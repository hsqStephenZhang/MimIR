#define _GNU_SOURCE
#include <dirent.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
__attribute__((destructor)) static void audit(void) {
 DIR *d=opendir("/proc/self/task"); struct dirent *e;
 if(!d)return;
 while((e=readdir(d))) {
  int tid=atoi(e->d_name); if(!tid)continue;
  cpu_set_t s; CPU_ZERO(&s);
  if(sched_getaffinity(tid,sizeof(s),&s))continue;
  fprintf(stderr,"tid %d cpus",tid);
  for(int i=0;i<20;i++)if(CPU_ISSET(i,&s))fprintf(stderr," %d",i);
  fprintf(stderr,"\n");
 }
 closedir(d);
}
