#ifndef FP_H
#define FP_H

#include<stdio.h>
#include<stdlib.h>
#include<layer.h>

typedef struct{
    int num_layers;
    layer** layers;
} network;

Matrix* layer_forward(layer* layer,Matrix* input);
Matrix* network_forward(network* nn,Matrix* network_input);
#endif 