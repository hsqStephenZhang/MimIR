# StableHLO Sort in MimIR

The `triu`/`tril` example demonstrates that MimIR can express both the type-level
shape contract and the implementation of a tensor operator in one composable
definition. StableHLO `sort` is a more demanding case study because its semantics
are split between two parts: the constraints on the tensor operands and the
semantics of the comparator supplied to the sort.

## Constraints and Implementation

A tensor operator has two major parts:

1. **Constraints.** These describe valid input and output types, shape invariants,
   and restrictions on attributes. For `sort`, the constraints include a non-empty
   variadic input list, identical input and output shapes, a common sort dimension,
   and a comparator whose argument types match every input element type.
2. **Implementation.** This describes how valid values are transformed. MimIR
   lowers `sort` to a stable bubble sort over every one-dimensional slice orthogonal
   to the selected dimension. All variadic operands are carried in one accumulator,
   so a single swap keeps keys and payload tensors synchronized.

The StableHLO dimension is a signed constant and may be negative. The importer
normalizes a negative dimension by adding the rank. The MimIR operator then uses
`Idx r`, which makes the normalized range constraint intrinsic to the type of the
dimension argument.

## Comparator Semantics

StableHLO `compare` selects its comparison type from the element type:

- signed integers use `SIGNED`;
- unsigned integers and booleans use `UNSIGNED`;
- floating-point values use `FLOAT` or `TOTALORDER`;
- complex values use `FLOAT` semantics for equality-style comparisons.

In an MLIR-style representation, these rules are usually checked in a verifier
separate from the comparator body. It is easy for a lowering to accept the right
function signature while accidentally using signed comparison for a boolean or an
unsigned integer. The `stablehlo_compare` implementation makes the policy explicit
through a type-indexed dictionary:

```mim
let %tensor.StableHloCompare = [
    T: *,
    compare_type: Nat,
    compare: [T, T] -> Bool,
];
```

The witness carries the element type, the StableHLO comparison policy, and the
primitive comparator together. A sort comparator can then be composed from
`%tensor.stablehlo_compare` rather than directly naming `%core.icmp` or
`%math.cmp`:

```mim
let I32_SIGNED =
    (I32, %tensor.compare_type.signed_int, %core.icmp.sl @i32);

lam less (lhs rhs: I32): Bool =
    %tensor.stablehlo_compare I32_SIGNED (lhs, rhs);
```

This is a typeclass-like design, but it does not require a general-purpose typeclass
solver. MimIR already uses explicit dictionaries for structures such as
`%tensor.Ring`. A future resolver can construct canonical witnesses from the
element type: `Bool` and unsigned integers resolve to unsigned comparison, signed
integers to signed comparison, and floating-point types to floating comparison.

## What Is Captured Today

The current implementation captures the following `sort` semantics:

- non-empty input count is checked during lowering;
- input and output shapes are the same by construction;
- all variadic operands are sorted in lockstep;
- the comparator argument order is exactly
  `lhs0, rhs0, lhs1, rhs1, ...`;
- negative dimensions are expected to be normalized by the importer;
- `Idx r` prevents an out-of-range normalized dimension;
- `is_stable = true` is implemented with a stable algorithm;
- `is_stable = false` may also return a stable result, which is permitted;
- signed integer and boolean unsigned comparison can be represented through
  explicit comparison witnesses.

The comparator's strict-weak-ordering contract is treated as a precondition. It is
not checked dynamically: validating transitivity for arbitrary values would require
an expensive, potentially unbounded set of calls and would change the cost model of
the operator.

## Remaining Extensions

The dictionary currently makes the comparison policy explicit, but users can still
manually construct an inconsistent tuple, such as an unsigned policy paired with a
signed comparator. A stronger type system could prevent this in two steps:

1. distinguish signed and unsigned integer types, rather than representing both only
   as an `Idx` carrier;
2. add compile-time instance resolution for canonical `StableHloCompare` witnesses.

That resolver would reject unsupported combinations during elaboration, while
partial evaluation would erase the witness and leave only the selected primitive
comparison. `TOTALORDER`, complex lexicographic comparison, and quantized
dequantize-compare require additional primitives before they can be represented
without weakening StableHLO semantics.
