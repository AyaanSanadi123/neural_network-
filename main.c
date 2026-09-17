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
void train_network(network* nn,Dataset* data,int epochs,int batch_size,Optimizer* opt,Matrix* (*loss_deriv_func)(Matrix*, Matrix*),ThreadPool* pool, BenchLogger* logger) {
   printf("--- STARTING TRAINING LOOP (%d Epochs, %d Samples, Batch Size: %d) ---\n", epochs, data->train_samples, batch_size);
    double target_range = data->target_max - data->target_min;
   Matrix* batch_input = create_matrix(data->num_features,batch_size);
   Matrix* batch_expected = create_matrix(data->target_features,batch_size);



    for(int epoch = 1 ; epoch <= epochs; epoch++){
        // initiate the logger at the beginning at each epoch 
        logger_start_epoch(logger);

        double train_mse = 0.0, train_mae = 0.0;
        double val_mse = 0.0, val_mae = 0.0;

        // time to chunk the dataset in batchs 
        for(int i = 0; i< data->train_samples;i+= batch_size){
            int current_batch_size = batch_size;

            // check for the edge case, if the last batch does not have the size of 1024
            if (i + batch_size > data -> train_samples)
            {
                current_batch_size = data->train_samples - i;
                batch_input->cols = current_batch_size;
                batch_expected->cols = current_batch_size;
                // this updates the last batch size to the size of the remaining samples
            }
            get_batch(data,data->train_inputs,data->train_targets,i,current_batch_size,batch_input,batch_expected);

            // time to run the forward pass 
            Matrix* prediction = network_forward(nn,batch_input,pool);
            // this now returns a 1*1024 matrix, 

            // now we de-normalise it, to get the values for the csv
            // just run a for loop from left to right and apply the reverse min-max formula 
            for (int b = 0; b < current_batch_size; b++) {
               double real_pred = (prediction->data[b] * target_range) + data->target_min;
                double real_target = (batch_expected->data[b] * target_range) + data->target_min;
                double error = real_pred - real_target;
                train_mse += (error * error);
                train_mae += fabs(error);
            }
            // backward pass 
            network_backward(nn,prediction,batch_expected,loss_deriv_func,pool);
            // update the weights 
            network_update_weights(nn,opt);

            network_free_caches(nn);
            free_matrix(prediction);
        }
        // Snap the columns back to the full batch size for the validation step
        batch_input->cols = batch_size;
        batch_expected->cols = batch_size;
        
        // validation step 
        for (int i = 0; i < data->val_samples; i += batch_size) {
            int current_batch_size = batch_size;
            
            if (i + batch_size > data->val_samples) {
                current_batch_size = data->val_samples - i;
                batch_input->cols = current_batch_size;
                batch_expected->cols = current_batch_size;
            }

            // Pass the VAL arrays into the universal shovel
            get_batch(data, data->val_inputs, data->val_targets, i, current_batch_size, batch_input, batch_expected);

            Matrix* predictions = network_forward(nn, batch_input, pool);
            
            for (int b = 0; b < current_batch_size; b++) {
                double real_pred = (predictions->data[b] * target_range) + data->target_min;
                double real_target = (batch_expected->data[b] * target_range) + data->target_min;
                double error = real_pred - real_target;
                val_mse += (error * error);
                val_mae += fabs(error);
            }

            // CRITICAL: No backward pass! No weight updates! Just free the memory.
            network_free_caches(nn);
            free_matrix(predictions);
        }
        
        batch_input->cols = batch_size;
        batch_expected->cols = batch_size;
        
        // Calculate the final averages
        train_mse /= data->train_samples;
        train_mae /= data->train_samples;
        val_mse /= data->val_samples;
        val_mae /= data->val_samples;
       
        printf("Epoch %d | Train MAE: %.4f | Val MAE: %.4f\n", epoch, train_mae, val_mae);
        logger_log_epoch(logger, epoch, train_mse, train_mae);
       
    } 
    // training ends 
    free_matrix(batch_input);
    free_matrix(batch_expected);
} 

void test_network(network* nn, Dataset* dataset, int batch_size, ThreadPool* pool) {
    printf("\n--- INITIATING FINAL BLIND TEST PHASE ---\n");
    
    double test_mse = 0.0;
    double test_mae = 0.0;
    double test_mape = 0.0; // MAPE accumulator
    double ss_res = 0.0;    // Residual sum of squares for R^2
    double ss_tot = 0.0;    // Total sum of squares for R^2
    
    double target_range = dataset->target_max - dataset->target_min;

    // STEP 1: Calculate the exact mean of the test dataset for R-Squared
    double test_target_mean = 0.0;
    for (int i = 0; i < dataset->test_samples; i++) {
        double real_target = (dataset->test_targets[i]->data[0] * target_range) + dataset->target_min;
        test_target_mean += real_target;
    }
    test_target_mean /= dataset->test_samples;

    Matrix* test_batch_input = create_matrix(dataset->num_features, batch_size);
    Matrix* test_batch_expected = create_matrix(dataset->target_features, batch_size);

    // STEP 2: The standard test sweep
    for (int i = 0; i < dataset->test_samples; i += batch_size) {
        int current_batch_size = batch_size;
        
        if (i + batch_size > dataset->test_samples) {
            current_batch_size = dataset->test_samples - i;
            test_batch_input->cols = current_batch_size;
            test_batch_expected->cols = current_batch_size;
        }

        get_batch(dataset, dataset->test_inputs, dataset->test_targets, i, current_batch_size, test_batch_input, test_batch_expected);

        Matrix* predictions = network_forward(nn, test_batch_input, pool);

        for (int b = 0; b < current_batch_size; b++) {
            double real_pred = (predictions->data[b] * target_range) + dataset->target_min;
            double real_target = (test_batch_expected->data[b] * target_range) + dataset->target_min;
            
            double error = real_pred - real_target;
            
            // Standard Error Metrics
            test_mse += (error * error);
            test_mae += fabs(error);
            
            // MAPE: Absolute Error divided by Actual Target
            if (real_target != 0.0) { // Safety check against division by zero
                test_mape += fabs(error / real_target);
            }
            
            // R-Squared Components
            ss_res += (error * error);
            ss_tot += (real_target - test_target_mean) * (real_target - test_target_mean);
        }

        network_free_caches(nn);
        free_matrix(predictions);
    }

    // Final Metric Averaging
    test_mse /= dataset->test_samples;
    test_mae /= dataset->test_samples;
    test_mape = (test_mape / dataset->test_samples) * 100.0; // Convert to percentage
    
    // R-Squared Formula
    double r_squared = 1.0 - (ss_res / ss_tot);

    printf("=========================================\n");
    printf("FINAL ENGINE GRADE ON UNSEEN TRACK DATA:\n");
    printf("Mean Squared Error: %.4f\n", test_mse);
    printf("Mean Absolute Error: %.4f ms off per lap\n", test_mae);
    printf("Mean Absolute Percentage Error (MAPE): %.2f%%\n", test_mape);
    printf("R-Squared (Knowledge Percentage): %.4f (%.2f%%)\n", r_squared, r_squared * 100.0);
    printf("=========================================\n\n");

    free_matrix(test_batch_input);
    free_matrix(test_batch_expected);
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
    ThreadPool* pool = thread_pool_init(8,1024);

    // create neural network of three layers 
    network* nn = create_network(3);
    nn -> layers[0] = create_layer(7,64,leaky_relu,leaky_relu_derivative);
    nn -> layers[1] = create_layer(64,32,leaky_relu,leaky_relu_derivative);
    nn -> layers[2] = create_layer(32,1,linear,linear_derivative);
   
    // initiate the optimizer 
    Optimizer* adam = create_adam_optimizer(0.002,nn->num_layers,nn->layers);
    int epochs = 100;
    int batch_size = 1024;
    train_network(nn,dataset,epochs,batch_size,adam,mse_derivative,pool,logger);

    // run the test after training to get the final accuracy 
    test_network(nn,dataset,batch_size,pool);

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




// windows 
// gcc -Iactivation_functions -Ilayers -Iloss_functions -Imatrix -Inetwork -Ioptimizer -Ithread_pool -Ibenchmarking_engine -Idata_preprocessing main.c activation_functions\activations.c layers\layer.c loss_functions\losses.c matrix\matrix.c network\fp.c optimizer\optimizer.c thread_pool\threadpool.c benchmarking_engine\b_engine.c data_preprocessing\data_loader.c -o neural_network.exe -pthread -lm

// mac 
// gcc -Iactivation_functions -Ilayers -Iloss_functions -Imatrix -Inetwork -Ioptimizer -Ithread_pool -Ibenchmarking_engine -Idata_preprocessing main.c activation_functions/activations.c layers/layer.c loss_functions/losses.c matrix/matrix.c network/fp.c optimizer/optimizer.c thread_pool/threadpool.c benchmarking_engine/b_engine.c data_preprocessing/data_loader.c -o neural_network -pthread -lm