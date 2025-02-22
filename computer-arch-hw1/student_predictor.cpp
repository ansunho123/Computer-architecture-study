#include "student_predictor.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

your_own::your_own()
{
  num_predictor_entry = 32000;
  pred_arr = new int[num_predictor_entry]{
      0,
  };
};
your_own::~your_own()
{
  if (pred_arr != NULL)
    delete[] pred_arr;
}
/* warning!!! Don't change argument of these function   */
int your_own::get_pred(int pc)
{
  int idx = pc % num_predictor_entry;
  int prediction = pred_arr[idx];
  if (prediction >= 2)
  {
    prediction = 1;
  }
  else
  {
    prediction = 0;
  }
  return prediction;
}
void your_own::update(int pc, int res)
{
  int idx = pc % num_predictor_entry;
  int *arr = pred_arr;
  int prediction = pred_arr[idx];
  if (res == 1)
  {
    if (prediction < 3)
    {
      arr[idx]++;
    }
    else
    {
      arr[idx] = 3;
    }
  }
  else
  {
    if (prediction != 0)
    {
      arr[idx]--;
    }
    else
    {
      arr[idx] = 0;
    }
  }
}
