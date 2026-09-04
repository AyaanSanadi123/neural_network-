#ifndef FP_H
#define FP_H

#include<stdio.h>
#include<stdlib.h>
#include<layer.h>
#include<losses.h>
#include<optimizer.h>
#include "threadpool.h"

typedef struct{
    int num_layers;
    layer** layers;
} network;

network* create_network(int num_layers);
Matrix* layer_forward(layer* layer,Matrix* input,ThreadPool* pool);
Matrix* network_forward(network* nn,Matrix* network_input,ThreadPool* pool);
Matrix* layer_backward(layer* l, Matrix* dA,ThreadPool* pool);
void network_backward(network* nn, Matrix* predictions, Matrix* expected, Matrix* (*loss_deriv_func)(Matrix*, Matrix*),ThreadPool* pool);
void network_update_weights(network* nn, Optimizer* opt);
void network_free_caches(network* nn);
void free_network(network* nn);
#endif 