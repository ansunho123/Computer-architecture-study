#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "student_predictor.hpp"

void prediction_ratio_update(int &count, int prediction, int true_value) {
  if (prediction == true_value) count++;
}
int main() {
  your_own predictor;
  FILE *branch_trace;
  char *token1;
  char strline[30];
  int correct = 0;
  int total = 0;
  int prediction = 0;
  int pc = 0;
  int dummy = 0;
  int branch;
  branch_trace = fopen("input1.csv", "rt");
  if (branch_trace == NULL) {
    printf("input file doesn'exist, terminate program");
    return 1;
  }
  while (fgets(strline, sizeof(strline), branch_trace)) {
    token1 = strtok(strline, " ,");
    pc = std::stoi(token1, nullptr, 16);
    token1 = strtok(NULL, " ,");
    branch = atoi(token1);
    token1 = strtok(NULL, "\n");
    dummy = atoi(token1);
    prediction = predictor.get_pred(pc);
    prediction_ratio_update(correct, prediction, branch);
    predictor.update(pc, branch);
    total++;
  }
  fclose(branch_trace);
  branch_trace = fopen("input2.csv", "rt");
  if (branch_trace == NULL) {
    printf("input file doesn'exist, terminate program");
    return 1;
  }
  while (fgets(strline, sizeof(strline), branch_trace)) {
    token1 = strtok(strline, " ,");
    pc = std::stoi(token1, nullptr, 16);
    token1 = strtok(NULL, " ,");
    branch = atoi(token1);
    token1 = strtok(NULL, "\n");
    dummy = atoi(token1);
    prediction = predictor.get_pred(pc);
    prediction_ratio_update(correct, prediction, branch);
    predictor.update(pc, branch);
    total++;
  }
  fclose(branch_trace);
  branch_trace = fopen("input3.csv", "rt");
  if (branch_trace == NULL) {
    printf("input file doesn'exist, terminate program");
    return 1;
  }
  while (fgets(strline, sizeof(strline), branch_trace)) {
    token1 = strtok(strline, " ,");
    pc = std::stoi(token1, nullptr, 16);
    token1 = strtok(NULL, " ,");
    branch = atoi(token1);
    token1 = strtok(NULL, "\n");
    dummy = atoi(token1);
    prediction = predictor.get_pred(pc);
    prediction_ratio_update(correct, prediction, branch);
    predictor.update(pc, branch);
    total++;
  }
  fclose(branch_trace);
  dummy = dummy+1;
  printf("total : %d branch, correct : %d , ratio : %f%%\n", total, correct,
      (float)correct / (float)total * 100.0);
  return 0;
}
