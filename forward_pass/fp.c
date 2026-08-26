#include<fp.h>


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
    return current_signal;
}