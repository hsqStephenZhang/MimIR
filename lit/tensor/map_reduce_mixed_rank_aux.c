#include <stdio.h>
#include <stdlib.h>

float* rows(float*, float*);

int main(void) {
    float x[32], s[4];
    for (int i = 0; i != 32; ++i) x[i] = (float)i;
    for (int i = 0; i != 4; ++i) s[i] = 100.0f * (float)i;
    float* y = rows(x, s);
    int bad  = 0;
    for (int i = 0; i != 4; ++i)
        for (int j = 0; j != 8; ++j) bad += y[i * 8 + j] != x[i * 8 + j] + s[i];
    free(y);
    printf("bad=%d\n", bad);
    return bad != 0;
}
