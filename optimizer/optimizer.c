#include<optimizer.h>
#include <stdlib.h>
#include<matrix.h>

// SGD
void sgd_update(layer* l, double lr){
    // 1. Scale the raw gradients by the learning rate: (lr * dW)
    Matrix* scaled_dW = matrix_multiply_scalar(l->d_weights, lr);
    Matrix* scaled_db = matrix_multiply_scalar(l->d_biases, lr);
    
    // 2. Subtract the scaled gradients from the original weights: W_new = W_old - (lr * dW)
    Matrix* new_weights = matrix_subtract(l->weights, scaled_dW);
    Matrix* new_biases  = matrix_subtract(l->biases, scaled_db);
    
    // 3. Free the old parameter memory before we overwrite the pointers
    free_matrix(l->weights);
    free_matrix(l->biases);
    
    // 4. Assign the newly updated matrices back to the layer
    l->weights = new_weights;
    l->biases = new_biases;
    
    // 5. Clean up the intermediate scaled matrices to prevent memory leaks
    free_matrix(scaled_dW);
    free_matrix(scaled_db);
}
Optimizer* create_sgd_optimizer(double learning_rate){
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->learning_rate = learning_rate;
    opt->update_func = sgd_update;
    return opt;
}


// ADAM 


void adam_update(Optimizer* opt, layer* l, int layer_index){
    AdamState* state = (AdamState*) opt -> state; // caste it from void to AdamState
    AdamLayerCache cache = state -> layer_caches[layer_index]; // get the cache values of the current layer 

    // get the global variables and populate the local variables 
    double lr = opt -> learning_rate;
    double b1 = state ->beta1;
    double b2 = state -> beta2;
    double eps = state -> epsilon;
    double t = (double) state -> t;
}


Optimizer* create_adam_optimizer(double learning_rate, int num_layers, layer** layers){
    // create the opt object 
    Optimizer* opt = (Optimizer*) malloc (sizeof(Optimizer));
    opt -> learning_rate = learning_rate;
    opt -> update_func = adam_update;
    // declare and populate the state
    AdamState* state = (AdamState*)malloc(sizeof(AdamState));
    state -> t = 0;
    state -> beta1 = 0.9;
    state -> beta2 = 0.999;
    state -> epsilon = 1e-8;

    state -> layer_caches = (AdamLayerCache*)malloc(num_layers * sizeof(AdamLayerCache));

    for (int i = 0; i < num_layers; i++)
    {
        int w_rows = layers[i] ->weights -> rows;
        int w_cols = layers[i] -> weights -> cols;

        int b_rows = layers[i] -> biases -> rows;
        int b_cols = layers[i] -> biases -> cols;

        state -> layer_caches[i].m_weights = create_matrix(w_rows,w_cols);
        state -> layer_caches[i].v_weights = create_matrix(w_rows,w_cols);

        state -> layer_caches[i].m_biases = create_matrix(b_rows,b_cols);
        state -> layer_caches[i].v_biases = create_matrix(b_rows,b_cols);

        // populated the cache layers with 0's
        for(int j = 0; j< w_rows * w_cols ; j++){
            state -> layer_caches[i].m_weights -> data[j] = 0.0;
            state -> layer_caches[i].v_weights -> data[j] = 0.0;
        }
        for(int j = 0; j< b_rows * b_cols ; j++){
            state -> layer_caches[i].m_biases -> data[j] = 0.0;
            state -> layer_caches[i].v_biases -> data[j] = 0.0;
        }
    }
    
    opt -> state = state;

    return opt;
}