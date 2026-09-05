#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "layer.h"
#include "matrix.h"

typedef struct {
    Matrix* m_weights;
    Matrix* v_weights;
    Matrix* m_biases;
    Matrix* v_biases;
}  AdamLayerCache;  


typedef struct {
    int t;
    double beta1;
    double beta2;
    double epsilon;
    AdamLayerCache* layer_caches;
}  AdamState;

typedef struct Optimizer Optimizer;

struct Optimizer{
    double learning_rate;
    void* state; // if a optimizer does not need any state, state = NULL

    void (*update_func)(Optimizer* opt,layer* l,int layer_index);
};


Optimizer* create_sgd_optimizer(double learning_rate);
void adam_update(Optimizer* opt, layer* l, int layer_index);
Optimizer* create_adam_optimizer(double learning_rate, int num_layers, layer** layers);
#endif