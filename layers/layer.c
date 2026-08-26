#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>

#include<layer.h>


double random_uniform(){
    return ((double)rand() / (double)RAND_MAX) - 0.5;
    // bw 0.5 and -0.5
}

layer* create_layer(int input_size,int output_size,double (*act_func)(double), double (*act_deriv)(double)){
    layer* l = (layer*)malloc(sizeof(layer));

    l -> input_size = input_size;
    l -> output_size = output_size;

    l -> weights = create_matrix(output_size,input_size);
    l -> biases = create_matrix(output_size,1);

    // the weights needs to be random (bw -0.5 and 0.5)
    for (int i = 0; i < l->weights->rows * l->weights->cols; i++)
    {
        l->weights->data[i] = random_uniform();
    }

    // we need to prepare cache for the forward pass, set it to null and update it for the forward pass 
    l -> input_cache = NULL;
    l -> z_cache = NULL;
    l -> activation_cache = NULL;

    // allocate space for graidents, for backward pass 
    l -> d_weights = create_matrix(output_size,input_size);
    l -> d_biases = create_matrix(output_size,1);

    l->activation_func = act_func;
    l->activation_derivative = act_deriv;

    return l;
}