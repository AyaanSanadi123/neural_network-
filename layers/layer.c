#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<activations.h>
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

    double limit = 0.0;
    if (act_func == relu || act_func == leaky_relu) {
        // He Initialization (Compensates for 50% data loss)
        limit = sqrt(2.0 / input_size);
    } else {
        // Xavier/Glorot Initialization (Standard variance preservation)
        limit = sqrt(1.0 / input_size);
    }

    for (int i = 0; i < l->weights->rows * l->weights->cols; i++) {
        // Generate random number between -1.0 and 1.0, then scale by the selected limit
        double rand_normalized = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
        l->weights->data[i] = rand_normalized * limit;
    }
    
    for (int i = 0; i < l->biases->rows * l->biases->cols; i++) {
        l->biases->data[i] = 0.0;
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

void layer_free_caches(layer* l){
    if (l->input_cache != NULL) {
        free_matrix(l->input_cache);
        l->input_cache = NULL;
    }
    
    if (l->z_cache != NULL) {
        free_matrix(l->z_cache);
        l->z_cache = NULL;
    }
    
    if (l->activation_cache != NULL) {
        free_matrix(l->activation_cache);
        l->activation_cache = NULL;
    }
}

void free_layer(layer* l){
    if(l->weights != NULL) free_matrix(l->weights);
    l->weights = NULL;

    if(l->biases != NULL) free_matrix(l->biases);
    l->biases = NULL;


    if(l->d_weights != NULL) free_matrix(l->d_weights); l->d_weights = NULL;
    if(l->d_biases != NULL) free_matrix(l->d_biases); l-> d_biases = NULL;

    layer_free_caches(l);
    free(l);
    
}