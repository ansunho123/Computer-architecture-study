#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "sem.h"
#include <x86intrin.h>
#include <immintrin.h>

static __inline__ uint64_t rdtsc(void)
{
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc"
                       : "=a"(lo), "=d"(hi));
  return ((uint64_t)lo | (((uint64_t)hi) << 32));
}

#define rMAX 16 // range of each element in the matrix

/* Students have to modify this function */
void OptimizedMatrixMultiplication(int **A,
                                   int **B,
                                   int **C,
                                   int size)
{
  for (uint32_t j = 0; j < size; j += 16)
  {
    for (uint32_t i = 0; i < size; i += 16)
    {
      for (uint32_t k = 0; k < size; k += 16)
      {
        for (uint32_t x = i; x < i + 16; x += 16)
        {
          for (uint32_t y = j; y < j + 16; y++)
          {
            __m512i c0 = _mm512_loadu_si512(&C[y][x]);
            for (uint32_t z = k; z < k + 16; z++)
            {
              __m512i b = _mm512_set1_epi32(B[y][z]);
              __m512i a = _mm512_loadu_si512(&A[z][x]);
              __m512i c1 = _mm512_mullo_epi32(a, b);
              c0 = _mm512_add_epi32(c0, c1);
            }
            _mm512_storeu_si512(&C[y][x], c0);
          }
        }
      }
    }
  }
}

/* Naive matrix multiplication function as a baseline */
uint64_t NaiveMatrixMultiplication(int **A,
                                   int **B,
                                   int **C,
                                   int size)
{
  uint64_t start_cycle, end_cycle;
  start_cycle = rdtsc(); // read the current cycle from cycle tsc register.

  for (uint32_t i = 0; i < size; i++)
  {
    for (uint32_t j = 0; j < size; j++)
    {
      for (uint32_t k = 0; k < size; k++)
      {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  end_cycle = rdtsc(); // read the current cycle from cycle tsc register.

  uint64_t exec_cycle = end_cycle - start_cycle;
  printf("Matrix Multiplication : Naive,     Execution cycles   = %ld \n", (exec_cycle));

  return exec_cycle;
}

// Used for checking results between your matrix multiplication and naive matrix multiplication.
void CorrectionCheck(int **C_ref, int **C, int size)
{
  int temp = 0;
  for (uint32_t i = 0; i < size; i++)
  {
    for (uint32_t j = 0; j < size; j++)
    {
      if (C_ref[i][j] == C[i][j])
        ;
      else
      {
        printf("wrong %d multiplication terminate program \n", size);
        printf("C : %d , C_your : %d\n", C_ref[i][j], C[i][j]);
        printf(" on (i,j) = (%d, %d)\n", i, j);
        temp += 1;
        continue;
      }
    }
  }
  if (temp == 0)
  {
    printf("Success!! \n\n");
  }
  else
  {
    printf("Total Error: %d\n\n", temp);
  }
}

uint64_t MatrixMultiplication(int **A,
                              int **B,
                              int **C,
                              int size)
{
  uint64_t start_cycle, end_cycle;

  start_cycle = rdtsc(); // read the current cycle from cycle tsc register.
  OptimizedMatrixMultiplication(A, B, C, size);
  end_cycle = rdtsc(); // read the current cycle from cycle tsc register.

  uint64_t exec_cycle = end_cycle - start_cycle;
  printf("Matrix Multiplication : Optimized, Execution cycles   = %ld \n", (end_cycle - start_cycle));

  return exec_cycle;
}

/* Insert random values into the matrix  */
void randomInit(int **data, int size)
{
  srand(0);

  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; j++)
    {
      data[i][j] = rand() % rMAX;
    }
  }
}

void zeroInit(int **data, int size)
{
  srand(0);

  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; j++)
    {
      data[i][j] = 0;
    }
  }
}

int main(int argc, char **argv)
{
  int matrix_size = 1024;
  char *option;
  if (argc < 2)
  {
    printf("Please insert arguments\n");
  }
  else
  {
    option = argv[1];
  }

  int semid = initsem(9486);
  p(semid);
  //-------------------Matrix Declarations--------------------//
  int **A = (int **)malloc(matrix_size * sizeof(int *));
  int **B = (int **)malloc(matrix_size * sizeof(int *));
  int **C = (int **)malloc(matrix_size * sizeof(int *));
  int **C_ref = (int **)malloc(matrix_size * sizeof(int *));

  for (int i = 0; i < matrix_size; i++)
  {
    A[i] = (int *)malloc(matrix_size * sizeof(int));
    B[i] = (int *)malloc(matrix_size * sizeof(int));
    C[i] = (int *)malloc(matrix_size * sizeof(int));
    C_ref[i] = (int *)malloc(matrix_size * sizeof(int));
  }

  //---Init---//
  randomInit(A, matrix_size);
  randomInit(B, matrix_size);
  zeroInit(C, matrix_size);
  zeroInit(C_ref, matrix_size);

  uint64_t exec_cycle_naive[10];
  uint64_t exec_cycle_optimized[10];

  uint64_t max_cycle_naive = 0;
  uint64_t min_cycle_naive = (uint64_t)-1;

  uint64_t max_cycle_optimized = 0;
  uint64_t min_cycle_optimized = (uint64_t)-1;

  uint64_t sum_exec_cycle_naive = 0;
  uint64_t sum_exec_cycle_optimized = 0;

  for (int i = 0; i < 6; i++)
  {
    //-----Run Baseline------//
    if (strcmp(option, "all") == 0)
    {
      exec_cycle_naive[i] = NaiveMatrixMultiplication(A, B, C_ref, matrix_size);
      if (exec_cycle_naive[i] > max_cycle_naive)
        max_cycle_naive = exec_cycle_naive[i];
      if (exec_cycle_naive[i] < min_cycle_naive)
        min_cycle_naive = exec_cycle_naive[i];
      sum_exec_cycle_naive += exec_cycle_naive[i];
    }

    //-----Run Optimized------//
    if (strcmp(option, "opt") == 0 || strcmp(option, "all") == 0)
    {
      exec_cycle_optimized[i] = MatrixMultiplication(A, B, C, matrix_size);
      if (exec_cycle_optimized[i] > max_cycle_optimized)
        max_cycle_optimized = exec_cycle_optimized[i];
      if (exec_cycle_optimized[i] < min_cycle_optimized)
        min_cycle_optimized = exec_cycle_optimized[i];
      sum_exec_cycle_optimized += exec_cycle_optimized[i];
    }
    //-----Functionaliry_Check-----//
    if (strcmp(option, "all") == 0)
    {
      CorrectionCheck(C_ref, C, matrix_size);
    }
  }

  sum_exec_cycle_naive -= max_cycle_naive;
  sum_exec_cycle_naive -= min_cycle_naive;

  sum_exec_cycle_optimized -= max_cycle_optimized;
  sum_exec_cycle_optimized -= min_cycle_optimized;

  printf("Matrix Multiplication done\n");
  printf("Naive Execution cycles     = %ld \n", (sum_exec_cycle_naive));
  printf("Optimized Execution cycles = %ld \n", (sum_exec_cycle_optimized));
  printf("Performance Improvement    = %.4f \n", (float)sum_exec_cycle_naive / (float)sum_exec_cycle_optimized);

  //-------------------Free--------------------//
  for (int i = 0; i < matrix_size; i++)
  {
    free(A[i]);
    free(B[i]);
    free(C[i]);
    free(C_ref[i]);
  }

  free(A);
  free(B);
  free(C);
  free(C_ref);

  v(semid);
  return 0;
}
