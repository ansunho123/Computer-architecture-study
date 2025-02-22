#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

class your_own
{
private:
  int num_predictor_entry;
  int *pred_arr;

public:
  your_own();
  your_own(int, int);
  ~your_own();
  int get_pred(int);
  void update(int, int);
};
