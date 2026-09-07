#include "data_loader.h"
#include<stdio.h>
#include<stdlib.h>


Dataset* create_dataset(int total_samples, int num_features, int target_features){
    Dataset* data = (Dataset*)malloc(sizeof(Dataset));

    data -> total_samples = total_samples;
    data -> num_features = num_features;
    data -> target_features = target_features;


    // allocate the blank matrices 
    data -> raw_inputs = (Matrix*) malloc(total_samples * sizeof(Matrix*));
    data -> raw_targets = (Matrix*) malloc(total_samples * sizeof(Matrix*));

    // allocate exaclty one matrix per row for both input and output 
    for (int i = 0; i < total_samples; i++)
    {
       data -> raw_inputs[i] = create_matrix(num_features,1);
       data -> raw_targets[i] = create_matrix(target_features,1);
    }
    data->train_inputs = NULL; data->train_targets = NULL;
    data->test_inputs = NULL;  data->test_targets = NULL;
    data->val_inputs = NULL;   data->val_targets = NULL;
    
    return data;

}


void split_dataset_sequential(Dataset* data, float train_ratio, float test_ratio, float val_ratio){
    // the goal is to break the dataset into test train and val sections 

    data -> train_samples = (int) (data -> total_samples * train_ratio);
    data -> test_samples = (int) (data -> total_samples * test_ratio);
    data -> val_samples = (int) (data -> total_samples * val_ratio);

    data->train_inputs = (Matrix**)malloc(data->train_samples * sizeof(Matrix*));
    data->train_targets = (Matrix**)malloc(data->train_samples * sizeof(Matrix*));
    
    data->test_inputs = (Matrix**)malloc(data->test_samples * sizeof(Matrix*));
    data->test_targets = (Matrix**)malloc(data->test_samples * sizeof(Matrix*));
    
    data->val_inputs = (Matrix**)malloc(data->val_samples * sizeof(Matrix*));
    data->val_targets = (Matrix**)malloc(data->val_samples * sizeof(Matrix*));

    int current_index = 0;

    for (int i = 0; i < data -> train_samples; i++,current_index++){
        data -> train_inputs[i] = data -> raw_inputs[current_index];
        data -> train_targets[i] = data -> raw_targets[current_index];
    }
    for(int i = 0; i < data->test_samples; i++, current_index++) {
        data->test_inputs[i] = data->raw_inputs[current_index];
        data->test_targets[i] = data->raw_targets[current_index];
    }
    
   
    for(int i = 0; i < data->val_samples; i++, current_index++) {
        data->val_inputs[i] = data->raw_inputs[current_index];
        data->val_targets[i] = data->raw_targets[current_index];
    }
    
}

void free_dataset(Dataset* data){
    for (int i = 0; i < data -> total_samples; i++)
    {
        free_matrix(data -> raw_inputs[i]);
        free_matrix(data -> raw_targets[i]);
    }
    free(data->raw_inputs);
    free(data -> raw_targets);

    if(data->train_inputs != NULL) {
        free(data->train_inputs); free(data->train_targets);
        free(data->test_inputs);  free(data->test_targets);
        free(data->val_inputs);   free(data->val_targets);
    }
    
    free(data);
}