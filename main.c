#include <stdio.h>
#include <stdlib.h>
#include "fp.h"
#include "layer.h"
#include "matrix.h"
#include "optimizer.h"
#include "losses.h"
#include "activations.h"
void train_network(network* nn, Matrix* input_data, Matrix* target_output, int epochs, Optimizer* opt, Matrix* (*loss_deriv_func)(Matrix*, Matrix*)) {
    printf("--- STARTING TRAINING LOOP (%d Epochs) ---\n", epochs);
    for (int epoch = 0; epoch < epochs; epoch++){
        
        // forward pass 
        Matrix* prediction = network_forward(nn,input_data);

        if (epoch % 10 == 0) {
            printf("Epoch %d - Prediction: %f (Target: %f)\n", 
                   epoch, prediction->data[0], target_output->data[0]);
        }

        // backward pass 
        network_backward(nn,prediction,target_output,loss_deriv_func);

        // update the weights 
        network_update_weights(nn,opt);

        // clean up the cache 
        network_free_caches(nn);
        free_matrix(prediction);
    }

}

int main(){

    // Create a network 
    network* nn = create_network(3);
    nn->layers[0] = create_layer(2,4,relu,relu_derivative);
    nn->layers[1] = create_layer(4, 4, relu, relu_derivative);
    nn->layers[2] = create_layer(4, 1, sigmoid, sigmoid_derivative);

    // initializer the optimizer 
    Optimizer* sgd = create_sgd_optimizer(0.01);


    // initiate the training loop 
    train_network(nn, input_data, target_output, 100, sgd, mse_derivative);

    // free all the pointers post training 
    free_network(nn);           // Our new destructor!
    free_matrix(input_data);
    free_matrix(target_output);
    free(sgd);
    printf("--- SHUTDOWN SUCCESSFUL ---\n");

    return 0;
}