#include<optimizer.h>
#include <stdlib.h>
#include<matrix.h>
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