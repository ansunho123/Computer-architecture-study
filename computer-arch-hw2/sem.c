#include "sem.h"

int initsem(key_t semkey) 
{
  union semun arg;
  int semid = semget(semkey, 1, 0666 | IPC_CREAT | IPC_EXCL);
  if (semid == -1) 
  {
    if (errno == EEXIST) semid = semget(semkey, 1, 0);
    else
    {
      perror("semget");
      exit(1);
    }
  } 
  else
  {
    arg.val = 8;
    int status = semctl(semid, 0, SETVAL, arg);
    if (status == -1)
    {
      perror("semctl - init");
      exit(1);
    }
  }
  return semid;
}

void p(int semid) 
{
	struct sembuf buf;
	buf.sem_num = 0;
	buf.sem_op = -1;
	buf.sem_flg = SEM_UNDO;
	if (semop(semid, &buf, 1) == -1) 
	{
		perror("semop - p");
		exit(1);
	}
}

void v(int semid) 
{
	struct sembuf buf;
	buf.sem_num = 0;
	buf.sem_op = 1;
	buf.sem_flg = SEM_UNDO;
	if (semop(semid, &buf, 1) == -1) 
	{
		perror("semop - v");
		exit(1);
	}
}
