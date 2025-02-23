#include <errno.h>
#include <stdio.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/sem.h>
#include <sys/types.h>
#include <unistd.h>

union semun
{
  int val;
  struct semid_ds *buf;
  unsigned short *array;
};

int initsem(key_t semkey);
void p(int semid);
void v(int semid);
