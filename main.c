#include <stdio.h>
#include <stdlib.h>
#include "fp.h"
#include "layer.h"
#include "matrix.h"
#include "optimizer.h"
#include "losses.h"
#include "activations.h"
#include "threadpool.h"
#include "b_engine.h"
#include "data_loader.h"
// Upgraded to accept arrays of matrices (Matrix**) and the number of flashcards (num_samples)
void train_network(network* nn,Dataset* data,int epochs,Optimizer* opt,Matrix* (*loss_deriv_func)(Matrix*, Matrix*),ThreadPool* pool, BenchLogger* logger) {
    printf("--- STARTING TRAINING LOOP (%d Epochs, %d Samples) ---\n", epochs, data->train_samples);
    double target_range = data->target_max - data->target_min;
   
    for(int epoch = 1 ; epoch <= epochs; epoch++){
        // initiate the logger at the beginning at each epoch 
        logger_start_epoch(logger);

        double epoch_mse = 0.0;
        double epoch_mae = 0.0;

        // start the core loop, this is where u pull out the train_inputs and run them through the network 
        // then at the end of the forward pass, get the error and run the back prop
        for (int i = 0; i < data->train_samples; i++)
        {
            // this does one complete forward pass through the network
            Matrix* prediction =  network_forward(nn,data->train_inputs[i],pool);
            
            // we are going to un-scale the prediction values to log them into training_benchmark
            // we are reversing the min-max formula we used to normalize them, refer data preprocessing 
            double real_pred = (prediction->data[0] * target_range) + data->target_min;
            double real_target = (data->train_targets[i]->data[0] * target_range) + data->target_min;


            double error = real_pred - real_target;
            epoch_mse += (error * error);
            epoch_mae += fabs(error);
            network_backward(nn, prediction, data->train_targets[i], loss_deriv_func, pool);
            network_update_weights(nn, opt);
            
            network_free_caches(nn);
            free_matrix(prediction);
        }
        
        epoch_mse /= data->train_samples;
        epoch_mae /= data->train_samples;

        logger_log_epoch(logger, epoch, epoch_mse, epoch_mae);
    } 
} 
int main(){
    const char* dataset_file = "data/lap_time_predictor_dataset_cleaned.csv";
    int rows,cols;

    // scan the current dataset 
    printf("Scanning dataset dimensions...\n");
    // dataset has header...
    count_csv_dimensions(dataset_file,&rows,&cols,1);
    printf("Found %d rows, %d columns.\n", rows, cols);
    // 7 input features and 1 target feature
    Dataset* dataset = create_dataset(rows,7,1);


    printf("Loading CSV into memory arenas...\n");
    load_csv(dataset_file, dataset, 1);

    // split the dataset
    split_dataset_sequential(dataset,0.7,0.2,0.1);

    // normalize the dataset 
    normalize_dataset(dataset);

    // this ends the data pre processing part 

    // lets initialize the engine modules 
    // first the benchmarking engine 
    BenchLogger* logger = logger_init("training_benchmark.csv");
    ThreadPool* pool = thread_pool_init(12,1024);

    // create neural network of three layers 
    network* nn = create_network(3);
    nn -> layers[0] = create_layer(7,64,relu,relu_derivative);
    nn -> layers[1] = create_layer(64,32,relu,relu_derivative);
    nn -> layers[2] = create_layer(32,1,linear,linear_derivative);
   
    // initiate the optimizer 
    Optimizer* adam = create_adam_optimizer(0.001,nn->num_layers,nn->layers);
    int epochs = 10;
    train_network(nn,dataset,epochs,adam,mse_derivative,pool,logger);



    // cleanup
    printf("--- CLEANING UP --- \n");
    free_optimizer(adam, nn->num_layers); 
    free_network(nn);           
    thread_pool_destroy(pool);
    logger_close(logger);
    free_dataset(dataset);
    
    printf("--- SHUTDOWN SUCCESSFUL ---\n");
    return 0;
}





// gcc -Iactivation_functions -Ilayers -Iloss_functions -Imatrix -Inetwork -Ioptimizer -Ithread_pool -Ibenchmarking_engine -Idata_preprocessing main.c activation_functions\activations.c layers\layer.c loss_functions\losses.c matrix\matrix.c network\fp.c optimizer\optimizer.c thread_pool\threadpool.c benchmarking_engine\b_engine.c data_preprocessing\data_loader.c -o neural_network.exe -pthread -lm