#include "data_loader.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<float.h>
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
    data->val_samples = data->total_samples - data->train_samples - data->test_samples;

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


// phase-1 (Measurement)
void count_csv_dimensions(const char* filepath, int* out_rows, int* out_cols){
    FILE* file = fopen(filepath,"r");

    if(file == NULL){
        printf("Fetal error, could not open file %s\n",filepath);
        exit(1);
    }
    int rows = 0;
    int commas = 0;
    int first_line_passed = 0;

    char buffer[65536] ; // this is 64kb of buffer in the L1 cache 
    size_t bytes_read;
    char last_ch = "\0";

    while((bytes_read = fread(buffer,1,sizeof(buffer),file) > 0)){
        for(size_t i = 0; i< bytes_read; i++){
            // get the charecter from the buffer 
            char ch = buffer[i];
            // now to count the number of commas, we just need to iterate the first row
            if(first_line_passed == 0 && ch == ',') commas++;

            if(ch == '\n'){
                rows++;
                first_line_passed = 1;
            }
            last_ch = ch;
        }
    }

    if(last_ch != '\n' && last_ch != '\0' && rows > 0) rows++;

    *out_rows = rows;
    *out_cols = commas + 1;

    fclose(file);
}

// the point of this function is to convert the data in the csv into raw numbers for the neural network 



void load_csv(const char* filepath, Dataset* dataset){
    FILE* file = fopen(filepath,"r");

    if(file == NULL){
        printf("Fetal error, could not open file %s\n",filepath);
        exit(1);
    }
    char line[4096]; // this is 4kbs of a line buffer, is usually enough for most datasets, if this is the bottle-neck for you, please update it 
    int row = 0;
    // the total cols = input features + output features 
    int total_cols = dataset -> num_features + dataset-> target_features;
    // use fgets to get one line at a time, 
    // here u take a line, clean it, put the numbers into the matrix and start with a new line
    // if your csv file has headers, uncomment this file to make a silent read, else it will just corrept the dataset
   // fgets(line, sizeof(line), file);
    while(fgets(line,sizeof(line),file)){

        line[strcspn(line, "\r\n")] = '\0';

        char* current = line; // this is our read head, that moves through the string, currently pointing at the first char in the line 
        for(int cols = 0; cols< total_cols; cols++){
            // this finds the first comma in the char array and returns it memory address
            char* comma_ptr = strchr(current,',');
            // if it doesnt find a comma, it returns NULL 

            if(comma_ptr != NULL) *comma_ptr = '\0';

            double val = 0.0;
            if(*current != '\0'){ // if there is a string and not \0 then get then convert the asci value into float64
                val = atof(current);
            }
            
            if(cols < dataset -> num_features){
                dataset -> raw_inputs[row] -> data[cols] = val;
            }else {
                dataset->raw_targets[row]->data[cols - dataset->num_features] = val;
            }

            if(comma_ptr != NULL) current = comma_ptr + 1;
        }
        row ++;
        if(row >= dataset -> total_samples) break;
    }
    fclose(file);
}

void normalize_dataset(Dataset* data){
    int features = data -> num_features;

    // create the arrays to hold the min and max values of each column 
    double* min_vals = (double*)malloc(features * sizeof(double));
    double* max_vals = (double*) malloc(features * sizeof(double));

    double inf = DBL_MAX;
    for (int i = 0; i < features; i++)
    {
        min_vals[i] = inf;
        max_vals[i] = -inf;
    }
    

    // time to scan only the training dataset and find the min and max values for each column
    for (int i = 0; i < data -> train_samples; i++)
    {
        for(int j = 0 ; j < features ; j++){
            double current_val = data -> train_inputs[i] -> data[j];

            if (current_val < min_vals[j]) min_vals[j] = current_val;
            if (current_val > max_vals[j]) max_vals[j] = current_val;
        }
    }


    // applying the normalization to train 

    for (int i = 0; i < data->train_samples; i++)
    {
       for(int j = 0; j < features; j++){
        double range = max_vals[j] - min_vals[j];
        if (range > 1e-7)
        {
           data -> train_inputs[i] -> data[j] = (data -> train_inputs[i] -> data[j] - min_vals[j]) / range;
        }else {
            data -> train_inputs[i] -> data [j] = 0.0;
        }
        
       }
    }
    
    // apply to test 
    for (int i = 0; i < data -> test_samples; i++)
    {
        for(int j = 0; j< features ; j++){
            double range = max_vals[j] - min_vals[j];
            if (range > 1e-7)
            {
                data -> test_inputs[i] -> data[j] = (data -> test_inputs[i] -> data[j] - min_vals[j]) / range;
            }
            else {
                data -> test_inputs[i] -> data[j] = 0.0;
            }
        }
    }
    // apply to val 
    for (int i = 0; i < data -> val_samples; i++)
    {
        for(int j = 0; j< features ; j++){
            double range = max_vals[j] - min_vals[j];
            if (range > 1e-7)
            {
                data -> val_inputs[i] -> data[j] = (data -> val_inputs[i] -> data[j] - min_vals[j]) / range;
            }
            else {
                data -> val_inputs[i] -> data[j] = 0.0;
            }
        }
    }
    
    free(min_vals);
    free(max_vals);

    // normalise the target values 
    data->target_min = DBL_MAX;
    data->target_max = -DBL_MAX;

    // scan the targets 
    for(int i = 0; i < data -> train_samples;i++){
        double current_rul = data -> train_targets[i] -> data[0]; // this assumes you just have one output column(for this version this is a limitation, will look into this later)

        if(current_rul < data -> target_min) data -> target_min = current_rul;
        if(current_rul > data -> target_max) data -> target_max = current_rul;
    }
    double target_range = data->target_max - data -> target_min;

    if(target_range > 1e-7){
        for (int i = 0; i < data->train_samples; i++) {
            data->train_targets[i]->data[0] = 
                (data->train_targets[i]->data[0] - data->target_min) / target_range;
        }
        
        // Scale Test Targets
        for (int i = 0; i < data->test_samples; i++) {
            data->test_targets[i]->data[0] = 
                (data->test_targets[i]->data[0] - data->target_min) / target_range;
        }
        
        // Scale Val Targets
        for (int i = 0; i < data->val_samples; i++) {
            data->val_targets[i]->data[0] = 
                (data->val_targets[i]->data[0] - data->target_min) / target_range;
        }
        
    }else {
        for (int i = 0; i < data->train_samples; i++) data->train_targets[i]->data[0] = 0.0;
        for (int i = 0; i < data->test_samples; i++) data->test_targets[i]->data[0] = 0.0;
        for (int i = 0; i < data->val_samples; i++) data->val_targets[i]->data[0] = 0.0;
    }


}