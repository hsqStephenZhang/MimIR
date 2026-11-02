#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

static _Thread_local int active;
static _Thread_local uint64_t calls, bytes, frees;
static _Thread_local struct timespec start;
extern void* __libc_malloc(size_t);
extern void* __libc_realloc(void*, size_t);
extern void* __libc_calloc(size_t, size_t);
extern void __libc_free(void*);
void* malloc(size_t n) {if(active){++calls;bytes+=n;} return __libc_malloc(n);}
void* calloc(size_t n,size_t m) {if(active){++calls;bytes+=n*m;} return __libc_calloc(n,m);}
void* realloc(void* p,size_t n) {if(active){++calls;bytes+=n;} return __libc_realloc(p,n);}
void free(void* p) {if(active)++frees;__libc_free(p);}
void bench_profile_begin(void) {calls=bytes=frees=0;clock_gettime(CLOCK_MONOTONIC,&start);active=1;}
void bench_profile_end(void) {
 struct timespec end;clock_gettime(CLOCK_MONOTONIC,&end);active=0;
 const char* path=getenv("PORT_PROFILE_JSON");if(!path)return;
 FILE* f=fopen(path,"a");if(!f)abort();
 fprintf(f,"{\"ns\":%llu,\"allocations\":%llu,\"bytes\":%llu,\"frees\":%llu}\n",
   (unsigned long long)((end.tv_sec-start.tv_sec)*1000000000LL+end.tv_nsec-start.tv_nsec),
   (unsigned long long)calls,(unsigned long long)bytes,(unsigned long long)frees);fclose(f);
}
