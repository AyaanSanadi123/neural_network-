#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "matrix.h"

typedef struct {
    // core dimensions 
    int num_features; // number of input columns
    int target_features; // number of output columns

    // the matrix allocation (this is the one time memory allocation)
    // it basically is an array of matrices 
    int total_samples;
    Matrix** raw_inputs;
    Matrix ** raw_targets;

    // this is pointers to train, 
    // this is not a new matrix array, it just points to the raw input and target arrays
    int train_samples;
    Matrix** train_inputs;
    Matrix** train_targets;

    // pointers for test 
    int test_samples;
    Matrix** test_inputs;
    Matrix** test_targets;

    // pointers for val 
    int val_samples;
    Matrix** val_inputs;
    Matrix** val_targets;

    // target scale 
    double target_min;
    double target_max;

} Dataset;


Dataset* create_dataset(int total_samples, int num_features, int target_features);
void split_dataset_sequential(Dataset* data, float train_ratio, float test_ratio, float val_ratio);
void free_dataset(Dataset* data);
void count_csv_dimensions(const char* filepath, int* out_rows, int* out_cols);
void load_csv(const char* filepath, Dataset* dataset);
void normalize_dataset(Dataset* data);
#endif