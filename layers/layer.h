#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<matrix.h>

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
} layer;


double random_uniform();
layer* create_layer(int input_size,int output_size);