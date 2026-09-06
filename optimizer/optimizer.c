#include<optimizer.h>
#include <stdlib.h>
#include<matrix.h>
#include <math.h>


// SGD
void sgd_update(Optimizer* opt,layer* l, int layer_index){
    double lr = opt -> learning_rate;
    int w_size = l -> weights -> rows * l -> weights -> cols;
    for (int i = 0; i < w_size; i++)
    {
        l -> weights -> data[i] -= lr * l -> d_weights -> data[i];
    }

    int b_size = l -> biases -> rows * l -> biases -> cols;

    for (int i = 0; i < b_size; i++)
    {
        l -> biases -> data[i] -= lr * l -> d_biases -> data[i];
    }
    
    
}

Optimizer* create_sgd_optimizer(double learning_rate){
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->learning_rate = learning_rate;
    opt->update_func = sgd_update;
    opt -> state = NULL;
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

    // pre calculate the bias correctios 
    double correct_m = 1.0 - pow(b1,t);
    double correct_v = 1.0 - pow(b2,t);

    int w_size = l->weights -> rows * l->weights -> cols;

    for(int i = 0; i < w_size; i++){
        double g = l-> d_weights -> data[i]; // this is the gt in maths 
        
        // calculate mt = b1 * m(t-1) + (1 - b1) * gt
        cache.m_weights -> data[i] = b1 * cache.m_weights -> data[i] + (1.0 - b1) * g;

        // calculate vt = b2* v(t-1) + (1 - b2) * gt**2
        cache.v_weights -> data[i] = b2 * cache.v_weights -> data[i] +(1.0 - b2) * (g*g);

        // apply the bias correction 
        double m_hat = cache.m_weights->data[i] / correct_m;
        double v_hat = cache.v_weights->data[i] / correct_v;


        // the final parameter upate 
        // Wt = Wt-1 - alpha * m_hat/ sqrt(v_hat) + eps 

        l -> weights -> data[i] -= lr * m_hat / (sqrt(v_hat) + eps);
    }

    // update the biases the same way 
    int b_size = l->biases->rows * l->biases->cols;
    for(int i = 0; i < b_size; i++) {
        double g = l->d_biases->data[i];
        
        cache.m_biases->data[i] = b1 * cache.m_biases->data[i] + (1.0 - b1) * g;
        cache.v_biases->data[i] = b2 * cache.v_biases->data[i] + (1.0 - b2) * (g * g);
        
        double m_hat = cache.m_biases->data[i] / correct_m;
        double v_hat = cache.v_biases->data[i] / correct_v;
        
        l->biases->data[i] -= lr * m_hat / (sqrt(v_hat) + eps);
    }

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


void free_optimizer(Optimizer* opt, int num_layers){
    if(opt -> state != NULL){
        AdamState* state = (AdamState*)opt -> state;
        for (int i = 0; i < num_layers; i++)
        {
            free_matrix(state -> layer_caches[i].m_weights);
            free_matrix(state -> layer_caches[i].v_weights);
            free_matrix(state -> layer_caches[i].m_biases);
            free_matrix(state -> layer_caches[i].v_biases);
        }
        free(state -> layer_caches);
        free(state);
    }
    free(opt);
}