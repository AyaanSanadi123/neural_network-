#include<fp.h>
#include<losses.h>
#include<optimizer.h>
#include<layer.h>
#include<stdio.h>
#include<math.h>

network* create_network(int num_layers){
    network* nn = (network*)malloc(sizeof(network));
    nn -> num_layers = num_layers;
    nn -> layers = (layer**)malloc(num_layers*sizeof(layer*));
    return nn;
}

Matrix* layer_forward(layer* layer, Matrix* input){
    // cache the input matrix weights 
    layer-> input_cache = matrix_copy(input);

    // W * X dot product of weight and input 
    Matrix* wx = dot_product(layer->weights,input);
    
    // prepare the preactivation (z = wx+b)
    // and cache this value aswell 
    layer->z_cache = matrix_add(wx,layer->biases);
    // apply the actiavtion and cache that aswell 
    layer->activation_cache = matrix_map(layer->z_cache,layer->activation_func);
    
    free_matrix(wx);
    return layer->activation_cache;
}

Matrix* network_forward(network* nn, Matrix* network_input){
    Matrix* current_signal = network_input;
     // this is layer->activation_cache of the previous layer, if its the first layer, then its the raw input itself

    for(int i = 0;i<nn->num_layers;i++){
        current_signal = layer_forward(nn->layers[i],current_signal);
    }
    return matrix_copy(current_signal);
}

Matrix* layer_backward(layer* l, Matrix* dA){
    // calculate dZ = dA ⊙ g'(Z)


    // first we need to get g'(z)
    Matrix * g_prime = matrix_map(l->z_cache,l->activation_derivative);
    // hadamard product of dA and g'(z)
    Matrix* dZ = hadamard_product(dA,g_prime);
    free_matrix(g_prime);

    // calculate dW = dZ * A_prev^T

    Matrix* A_prev_T = transpose(l->input_cache);
    Matrix* new_d_weights = dot_product(dZ,A_prev_T);

    if (l->d_weights != NULL) free_matrix(l->d_weights);
    l->d_weights = new_d_weights;

    free_matrix(A_prev_T);

    // calculate db = dZ
    if (l->d_biases != NULL) free_matrix(l->d_biases);
    l->d_biases = matrix_copy(dZ);

    // calculate dA_prev = W^T * dZ
    Matrix* W_T = transpose(l->weights);
    Matrix* dA_prev = dot_product(W_T, dZ);
    free_matrix(W_T);
    free_matrix(dZ);
    return dA_prev;
}

void network_backward(network* nn, Matrix* predictions, Matrix* expected, Matrix* (*loss_deriv_func)(Matrix*, Matrix*)){
    // get the initial error, by running the loss function on the result 
    Matrix* current_error = loss_deriv_func(predictions,expected);

    // now run this backwards through the nework 
    for(int i = nn->num_layers - 1;i >= 0;i--){
        // updates dW,dB and resturns the error for the previous layer
        Matrix* error_for_prev_layer = layer_backward(nn->layers[i],current_error);

        free_matrix(current_error);
        current_error = error_for_prev_layer;
    }
    free_matrix(current_error);
}


void network_update_weights(network* nn, Optimizer* opt){
    for (int i = 0; i < nn->num_layers; i++)
    {
        opt->update_func(nn->layers[i],opt->learning_rate);
    }
    
}

void network_free_caches(network* nn){
    for (int i = 0; i < nn->num_layers; i++)
    {
         layer_free_caches(nn->layers[i]);
    }
    
}

void free_network(network* nn){
    for (int i = 0; i < nn->num_layers; i++)
    {
        free_layer(nn->layers[i]);
    }
    // free the array
    free(nn->layers);

    // free the network struct 
    free(nn);
}