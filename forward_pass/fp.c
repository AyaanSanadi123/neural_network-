#include<fp.h>


#include<stdio.h>
#include<math.h>


Matrix* layer_forward(layer* layer, Matrix* input){
    // cache the input matrix weights 
    layer-> input_cache = matrix_copy(input);

    // W * X dot product of weight and input 
    Matrix* wx = dot_product(layer->weights,input);
    
    // prepare the preactivation (z = wx+b)
    // and cache this value aswell 
    layer->z_cache = matrix_add(wx,layer->biases);
    // apply the actiavtion and cache that aswell 
    layer->activation_cache = matrix_map(layer->z_cache,layer->activation_cache);
    
    free_matrix(wx);
    return layer->activation_cache;
}

Matrix* network_forward(network* nn, Matrix* network_input){
    Matrix* current_signal = network_input;

    for(int i = 0;i<nn->num_layers;i++){
        current_signal = layer_forward(nn->layers[i],current_signal);
    }
    return current_signal;
}