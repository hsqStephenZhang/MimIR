# Map-Reduce Fusion in Compiler Terms

This note describes map-reduce fusion using Python-like pseudocode and index functions. The central question is not only whether one expression can be substituted into another, but whether the compiler can remove the intermediate tensor without changing reduction order or duplicating expensive work.

## 1. A compiler-oriented map-reduce model

A tensor `map_reduce` can be viewed as the following function:

```python
def map_reduce(
    output_shape,
    reduction_shape,
    inputs,
    input_maps,
    map_fn,
    reduce_fn,
    init,
    output_map=lambda output_index: output_index,
):
    output = Tensor(output_shape)

    for output_index in indices(output_shape):
        accumulator = init

        for reduction_index in indices(reduction_shape(output_index)):
            elements = [
                tensor[input_map(output_index, reduction_index)]
                for tensor, input_map in zip(inputs, input_maps)
            ]

            value = map_fn(*elements)
            accumulator = reduce_fn(accumulator, value)

        output[output_map(output_index)] = accumulator

    return output
```

The important pieces are:

- `output_index`: identifies an independent output element;
- `reduction_index`: visits the values contributing to that output element;
- `input_map(output_index, reduction_index)`: maps loop indices to an input tensor index;
- `output_map(output_index)`: maps a parallel loop index to an output tensor index;
- `map_fn`: computes one reduction contribution;
- `reduce_fn`: combines contributions into an accumulator.

Different `output_index` iterations can run in parallel when `output_map` does not map two iterations to the same output element.

Parallelizing the reduction loop requires stronger algebraic properties. A tree reduction normally requires `reduce_fn` to be associative. Arbitrary reordering also requires commutativity. Floating-point addition is not strictly associative, so the compiler must follow the operator's reassociation and precision rules.

## 2. Two connected map-reduces

Consider a producer and consumer:

```python
producer = map_reduce(...)
consumer = map_reduce(inputs=[producer, *other_inputs], ...)
```

The consumer reads the producer through an index function:

```python
producer_index = consumer_to_producer_index(
    consumer_output_index,
    consumer_reduction_index,
)

producer_value = producer[producer_index]
```

Semantic inlining replaces `producer[producer_index]` with the computation that produces that element. This is usually valid for pure operators.

However, three different transformations must be distinguished:

1. **Semantic inlining**: substitute the producer expression into the consumer.
2. **Nested-loop fusion**: compute the required producer value inside the consumer loop without materializing the full producer tensor.
3. **Flat reduction fusion**: merge both reduction loops into one reduction.

Each transformation requires stronger conditions than the previous one.

## 3. Fusing a map producer

A map producer has no reduction loop:

```python
def producer_element(producer_output_index):
    elements = [
        tensor[input_map(producer_output_index)]
        for tensor, input_map in zip(inputs, producer_input_maps)
    ]
    return producer_map_fn(*elements)
```

Suppose the consumer reads it as follows:

```python
producer_index = consumer_to_producer_index(
    consumer_output_index,
    consumer_reduction_index,
)
value = producer[producer_index]
```

Fusion replaces the tensor read with a scalar call:

```python
producer_index = consumer_to_producer_index(
    consumer_output_index,
    consumer_reduction_index,
)
value = producer_element(producer_index)
```

The fused input index function is ordinary function composition:

```python
def fused_input_map(consumer_output_index, consumer_reduction_index):
    producer_output_index = consumer_to_producer_index(
        consumer_output_index,
        consumer_reduction_index,
    )
    return producer_input_map(producer_output_index)
```

For example:

```python
activated[i, k] = relu(x[i, k])

output[i, j] = sum(
    activated[i, k] * weight[k, j]
    for k in range(K)
)
```

becomes:

```python
output[i, j] = sum(
    relu(x[i, k]) * weight[k, j]
    for k in range(K)
)
```

### Conditions used by MimIR today

MimIR currently applies this scalar inlining when:

- the producer has no reduction dimensions: `producer.Rr == 0`;
- the producer loop shape equals its output shape: `producer.Sr == producer.So`;
- the producer output map is the identity;
- producer and consumer belong to the same fusion group;
- the producer has one consumer, so inlining does not duplicate work.

These conditions ensure that every requested producer element is one scalar computation whose input maps can be composed with the consumer access map.

## 4. Fusing a reduction into a pointwise epilogue

Consider matrix multiplication followed by bias and ReLU:

```python
matmul[i, j] = sum(
    lhs[i, k] * rhs[k, j]
    for k in range(K)
)

biased[i, j] = matmul[i, j] + bias[i, j]
output[i, j] = relu(biased[i, j])
```

The fused loop is:

```python
for i in range(M):
    for j in range(N):
        accumulator = 0.0

        for k in range(K):
            accumulator += lhs[i, k] * rhs[k, j]

        value = accumulator + bias[i, j]
        output[i, j] = relu(value)
```

The epilogue runs after the reduction finishes and immediately before the output store.

The following transformation is generally wrong:

```python
# Wrong: bias and ReLU execute once per reduction iteration.
for k in range(K):
    accumulator += relu(lhs[i, k] * rhs[k, j] + bias[i, j])
```

In general:

```python
relu(sum(values)) != sum(relu(value) for value in values)
sum(values) + bias != sum(value + bias for value in values)
```

### Conditions used by MimIR today

MimIR's `%tensor.map_reduce_epilogue` currently requires:

- the consumer is pointwise and has no reduction dimensions;
- the consumer output map is the identity;
- consumer and reduction anchor have the same output shape and rank;
- the consumer reads the anchor with the identity input map;
- one epilogue group contains one complex reduction anchor;
- additional tensors, such as bias, are explicit epilogue inputs with their own index maps.

If two matrix multiplications meet at an add, they cannot become one contraction. A valid strategy is:

```python
right = materialize(matmul(a1, b1))
output = matmul_with_epilogue(
    a0,
    b0,
    epilogue=lambda left, index: left + right[index],
)
```

One matmul is materialized. The other matmul owns the add epilogue.

## 5. Broadcast and reuse

Pointwise epilogue fusion is simplest when one producer output is consumed by one consumer output:

```python
producer_index = consumer_output_index
```

A broadcast may reuse one producer element for many consumer elements:

```python
producer_index = (consumer_output_index.column,)
```

The compiler can count this reuse as:

```python
def reuse_count(producer_index):
    return count(
        consumer_index
        for consumer_index in indices(consumer_shape)
        if consumer_to_producer_index(consumer_index) == producer_index
    )
```

If `reuse_count(producer_index) > 1`, directly inlining a reduction producer may recompute the same reduction several times. Efficient fusion then needs one of:

- an intermediate tensor;
- a local or shared-memory cache;
- a loop schedule that keeps the producer result live while processing all its consumers.

Map producers are often cheap enough to duplicate. Reduction producers usually are not.

## 6. Fusing two reductions as nested loops

Suppose a consumer reduction reads a producer reduction:

```python
producer[u] = reduce(
    producer_map(u, producer_reduction_index)
    for producer_reduction_index in producer_reduction_domain(u)
)

consumer[v] = reduce(
    consumer_map(
        v,
        consumer_reduction_index,
        producer[
            consumer_to_producer_index(v, consumer_reduction_index)
        ],
    )
    for consumer_reduction_index in consumer_reduction_domain(v)
)
```

The producer reduction must normally finish before the consumer can use its result. A materialization-free nested loop is:

```python
for consumer_output_index in indices(consumer_output_shape):
    outer_accumulator = consumer_init

    for consumer_reduction_index in indices(
        consumer_reduction_shape(consumer_output_index)
    ):
        producer_output_index = consumer_to_producer_index(
            consumer_output_index,
            consumer_reduction_index,
        )

        inner_accumulator = producer_init

        for producer_reduction_index in indices(
            producer_reduction_shape(producer_output_index)
        ):
            producer_inputs = [
                tensor[
                    input_map(
                        producer_output_index,
                        producer_reduction_index,
                    )
                ]
                for tensor, input_map in producer_inputs_and_maps
            ]

            inner_value = producer_map_fn(*producer_inputs)
            inner_accumulator = producer_reduce_fn(
                inner_accumulator,
                inner_value,
            )

        outer_value = consumer_map_fn(
            consumer_output_index,
            consumer_reduction_index,
            inner_accumulator,
        )
        outer_accumulator = consumer_reduce_fn(
            outer_accumulator,
            outer_value,
        )

    output[consumer_output_index] = outer_accumulator
```

This uses a scalar `inner_accumulator` instead of a full producer tensor.

### Conditions for nested reduction fusion

This transformation is practical when:

1. `consumer_to_producer_index(...)` is available before the inner reduction starts;
2. producer input maps can be called with that producer index;
3. each required producer value is computed once, or recomputation is acceptable;
4. the producer has no unfused external consumers;
5. the loop schedule preserves the boundary between inner and outer reduction;
6. the new reduction order is allowed by the operator's floating-point semantics.

A useful no-recomputation check is:

```python
for producer_index in producer_output_indices:
    users = [
        (consumer_output_index, consumer_reduction_index)
        for consumer_output_index in consumer_output_indices
        for consumer_reduction_index in consumer_reduction_indices(
            consumer_output_index
        )
        if consumer_to_producer_index(
            consumer_output_index,
            consumer_reduction_index,
        ) == producer_index
    ]

    require(len(users) <= 1)
```

When this check fails, the implementation needs caching, materialization, or an explicit decision to recompute.

MimIR does not currently implement this hierarchical reduction-to-reduction fusion.

## 7. When two reductions can be flattened

Nested-loop fusion keeps two accumulators and two reduction levels. Flattening removes the boundary and uses one reduction.

This is valid only for compatible reduction functions.

For example, two sums can be flattened:

```python
result = sum(
    sum(x[i][j] for j in range(J))
    for i in range(I)
)
```

becomes:

```python
result = sum(
    x[i][j]
    for i in range(I)
    for j in range(J)
)
```

The transformation works because the inner and outer operations are the same associative reduction.

A more general version is possible when the function between the reductions preserves the reduction operation:

```python
transform(combine_a(x, y)) == combine_b(transform(x), transform(y))
transform(identity_a) == identity_b
```

This is the compiler-facing form of a monoid homomorphism.

The following example cannot be flattened in this way:

```python
result = sum(
    exp(max(x[i][j] for j in range(J)))
    for i in range(I)
)
```

`exp(max(x, y))` is not equal to `exp(x) + exp(y)`. The inner `max` must finish before `exp` and the outer `sum`.

## 8. Legal fusion may still be unprofitable

Consider row normalization:

```python
row_sum[i] = sum(x[i][k] for k in range(K))
output[i][j] = x[i][j] / row_sum[i]
```

Naive inlining produces:

```python
output[i][j] = x[i][j] / sum(x[i][k] for k in range(K))
```

This recomputes the same row reduction for every `j`. The substitution is semantically valid but usually unprofitable.

Efficient implementations need materialization or caching:

```python
for i in range(I):
    cached_row_sum = sum(x[i][k] for k in range(K))

    for j in range(J):
        output[i][j] = x[i][j] / cached_row_sum
```

Fusion legality and profitability must therefore be separate decisions.

## 9. Summary

The practical hierarchy is:

```text
semantic inlining
    contains nested-loop fusion
        contains flat reduction fusion
```

- **Semantic inlining** requires pure computation and valid index substitution.
- **Nested-loop fusion** additionally requires a legal schedule and manageable reuse.
- **Flat reduction fusion** additionally requires compatible reduction functions.

MimIR currently supports:

```text
map producer -> map or reduction consumer
reduction producer -> pointwise epilogue
```

MimIR does not yet support general:

```text
reduction producer -> reduction consumer
```

A future `hierarchical_map_reduce` could make the following concepts explicit:

- parallel/output indices;
- inner and outer reduction indices;
- epilogues between reduction levels;
- producer-result reuse;
- local, shared, or global memory scope;
- target-specific loop scheduling.
