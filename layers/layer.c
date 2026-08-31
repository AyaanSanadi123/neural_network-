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