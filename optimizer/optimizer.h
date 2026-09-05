#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "layer.h"
#include "matrix.h"

typedef struct {
    double learning_rate;
    void(*update_func)(layer* l,double lr);
}  Optimizer;

Optimizer* create_sgd_optimizer(double learning_rate);
#endif