#include <stdio.h>
#include <stdlib.h>
#include "fp.h"
#include "layer.h"
#include "matrix.h"
#include "optimizer.h"
#include "losses.h"
#include "activations.h"
#include "threadpool.h"
// Upgraded to accept arrays of matrices (Matrix**) and the number of flashcards (num_samples)
void train_network(network* nn, Matrix** input_data, Matrix** target_output, int num_samples, int epochs, Optimizer* opt, Matrix* (*loss_deriv_func)(Matrix*, Matrix*),ThreadPool* pool) {
    
    printf("--- STARTING TRAINING LOOP (%d Epochs) ---\n", epochs);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        
        // ====================================================
        // 1. TRAINING PHASE (Loop through every flashcard)
        // ====================================================
        for (int i = 0; i < num_samples; i++) {
            Matrix* prediction = network_forward(nn, input_data[i],pool);
            network_backward(nn, prediction, target_output[i], loss_deriv_func,pool);
            network_update_weights(nn, opt);
            
            network_free_caches(nn);
            free_matrix(prediction);
        }

        // ====================================================
        // 2. LOGGING PHASE (Print progress every 1000 epochs)
        // ====================================================
        if (epoch % 1000 == 0) {
            printf("\n--- Epoch %d ---\n", epoch);
            for(int i = 0; i < num_samples; i++) {
                Matrix* pred = network_forward(nn, input_data[i],pool);
                printf("Input [%.1f, %.1f] -> Predicted: %.4f (Target: %.1f)\n", 
                       input_data[i]->data[0], input_data[i]->data[1], 
                       pred->data[0], target_output[i]->data[0]);
                
                network_free_caches(nn);
                free_matrix(pred);
            }
        }
        
    } 
    
} 
int main(){

    // Create a network 
   // 1. Create the Architecture (Input: 2 -> Hidden: 8 -> Output: 1)
    network* nn = create_network(2);
    // Hidden layer needs a non-linear activation like ReLU to solve XOR
    nn->layers[0] = create_layer(2, 8, relu, relu_derivative); 
    // Output layer uses Sigmoid to squash the final answer between 0 and 1
    nn->layers[1] = create_layer(8, 1, sigmoid, sigmoid_derivative);

    Optimizer* sgd = create_adam_optimizer(0.01,nn->num_layers,nn ->layers); // Learning rate of 0.1

    ThreadPool* pool = thread_pool_init(4,10);
    // 2. Prepare the XOR Dataset (4 inputs, 4 targets)
    Matrix* inputs[4];
    Matrix* targets[4];

    for(int i = 0; i < 4; i++) {
        inputs[i] = create_matrix(2, 1);
        targets[i] = create_matrix(1, 1);
    }

    // Flashcard 1: [0, 0] -> [0]
    inputs[0]->data[0] = 0.0; inputs[0]->data[1] = 0.0; targets[0]->data[0] = 0.0;
    // Flashcard 2: [0, 1] -> [1]
    inputs[1]->data[0] = 0.0; inputs[1]->data[1] = 1.0; targets[1]->data[0] = 1.0;
    // Flashcard 3: [1, 0] -> [1]
    inputs[2]->data[0] = 1.0; inputs[2]->data[1] = 0.0; targets[2]->data[0] = 1.0;
    // Flashcard 4: [1, 1] -> [0]
    inputs[3]->data[0] = 1.0; inputs[3]->data[1] = 1.0; targets[3]->data[0] = 0.0;

    // 3. The Custom Training Loop
    printf("--- STARTING TRAINING ---\n");
    int epochs = 5000;

    // initiate the training loop 
    train_network(nn, inputs, targets, 4,epochs, sgd, mse_derivative,pool);

    // free all the pointers post training
    free_optimizer(sgd,nn->num_layers); 
    free_network(nn);           // Our new destructor!
   for(int i = 0; i < 4; i++) {
        free_matrix(inputs[i]);
        free_matrix(targets[i]);
    }
    
    thread_pool_destroy(pool);
    printf("--- SHUTDOWN SUCCESSFUL ---\n");

    return 0;
}





// gcc -Iactivation_functions -Ilayers -Iloss_functions -Imatrix -Inetwork -Ioptimizer -Ithread_pool main.c activation_functions\activations.c layers\layer.c loss_functions\losses.c matrix\matrix.c network\fp.c optimizer\optimizer.c thread_pool\threadpool.c -o neural_network.exe -pthread