#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Node {
    float acc[16];
    struct Node* next;
} Node;

typedef struct Padded {
    uint8_t tag;
    float acc[16];
    struct Padded* next;
    uint8_t tail;
} Padded;

extern Node* copy_node(Node*);
extern Padded* copy_padded(Padded*);

int main(void) {
    _Static_assert(sizeof(Node) == 72, "Node layout");
    _Static_assert(sizeof(Padded) == 88, "Padded layout");
    Node n   = {.next = &n};
    Padded p = {.tag = 23, .next = &p, .tail = 91};
    for (int i = 0; i < 16; ++i) {
        n.acc[i] = (float)i;
        p.acc[i] = (float)(2 * i);
    }
    Node* q   = copy_node(&n);
    Padded* r = copy_padded(&p);
    assert(q->next == &n && r->next == &p);
    assert(r->tag == 23 && r->tail == 92);
    for (int i = 0; i < 16; ++i) {
        assert(q->acc[i] == n.acc[i] + (i == 0));
        assert(r->acc[i] == p.acc[i] + (i == 0));
    }
    free(q);
    free(r);
    return 0;
}
