#ifndef LAYER_H 
#define LAYER_H


#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<matrix.h>
#include "threadpool.h"

typedef struct{
    int input_size;
    int output_size;

    Matrix* weights;
    Matrix* biases;

    Matrix* input_cache;
    Matrix* z_cache;
    Matrix* activation_cache;

    Matrix* d_weights;
    Matrix* d_biases;

    double (*activation_func)(double);
    double (*activation_derivative)(double);
} layer;


double random_uniform();
layer* create_layer(int input_size,int output_size,double (*act_func)(double), double (*act_deriv)(double));
void layer_free_caches(layer* l);
void free_layer(layer* l);
#endif