#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<assert.h>

#include<matrix.h>

// this is the payload sent to the worker threads 
typedef struct {
    Matrix* a;
    Matrix * b;
    Matrix* c;
    int start_row;
    int end_row;
} DotTask;

// the goal of this function, is to do the maths just for its assigned rows
static void dot_worker(void * arg){
    DotTask* task = (DotTask*) arg;

    for(int i = task -> start_row; i< task ->end_row;i++){
        for(int j = 0 ; j < task -> b -> cols; j++){
            double sum = 0.0;
            for(int k = 0 ; k < task -> a -> cols ; k ++){
                int index_a = i * task -> a -> cols + k;
                int index_b = k * task -> b -> cols + j;

                sum += task -> a -> data[index_a] * task -> b -> data[index_b];
            }
            task->c->data[i * task->c->cols + j] = sum;
        }
    }
}
// Constructor 

Matrix * create_matrix(int rows,int cols ){
    Matrix * m = (Matrix*)malloc(sizeof(Matrix));
    m -> rows = rows;
    m -> cols = cols;
    m-> data = (double*)calloc(rows * cols, sizeof(double));
    return m;
}

// destructor 

void free_matrix(Matrix * m){
    if( m != NULL){
        free(m->data);
        free(m);
    }
}


// dot product 

Matrix * dot_product(Matrix* a,Matrix * b,ThreadPool * pool){
    // safelty check 
    assert(a -> cols == b -> rows);
    // create a matrix 
    Matrix * c = create_matrix(a-> rows,b-> cols);

    // if no multi threading is required 
    if (pool == NULL) {
        for (int i = 0; i < a->rows; i++) {
            for (int j = 0; j < b->cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < a->cols; k++) {
                    sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
                }
                c->data[i * c->cols + j] = sum;
            }
        }
        return c;
    }

    // if multi threading is being used 

    int actual_threads;

    if(a->rows < pool -> num_threads){
        // each row gets its own thread
        actual_threads = a -> rows;
    }
    else {
        // if, we have more rows than threads 
        // wakeup every thread 
        actual_threads = pool -> num_threads;
    }

    // number of rows per thread 
    int rows_per_thread = a->rows / actual_threads;


    

}

// transpose a matrix 

Matrix* transpose(Matrix * m){
    Matrix * t = create_matrix(m->cols,m->rows);

    for (int i = 0; i < m->rows; i++)
    {
        for(int j = 0;j<m-> cols;j++){
            t->data[j*t->cols+i] = m->data[i * m->cols + j];
        }
    }
    return t;
}

// hadamard product 

Matrix * hadamard_product(Matrix* a,Matrix* b){
    assert(a->rows == b -> rows && a->cols == b-> cols);


    Matrix* c = create_matrix(a->rows,a->cols);

    int total_elements = a->rows*a->cols;
    for (int i = 0; i < total_elements; i++)
    {
       c->data[i] = a->data[i] * b->data[i];
    }
    return c;
}

Matrix* matrix_copy(Matrix* m){
    Matrix* copy = create_matrix(m->rows,m->cols);

    int total_elements = m->rows*m->cols;
    for (int i = 0; i < total_elements; i++)
    {
        copy->data[i] = m->data[i];
    }
    return copy;
}

Matrix* matrix_add(Matrix* a,Matrix* b){
    assert(a->rows == b->rows && a->cols == b->cols);

    Matrix* c = create_matrix(a->rows,a->cols);
    int total_elements = a->rows * a->cols;
    for (int i = 0; i < total_elements; i++)
    {
       c->data[i] = a->data[i] + b->data[i];
    }
    return c;
}

Matrix* matrix_map(Matrix* m, double (*func)(double)) {
    Matrix* result = create_matrix(m->rows, m->cols);
    
    int total_elements = m->rows * m->cols;
    for (int i = 0; i < total_elements; i++) {
        result->data[i] = func(m->data[i]);
    }   
    return result;
}
Matrix* matrix_subtract(Matrix* a, Matrix* b){
    assert(a->rows == b-> rows && a->cols == b->cols);

    Matrix* c = create_matrix(a->rows,a->cols);

    int total_elements = a->rows * a-> cols;
    for (int i = 0; i < total_elements; i++)
    {
       c->data[i] = a->data[i] - b ->data[i];
    }
    return c;
}

Matrix* matrix_multiply_scalar(Matrix* m, double scalar){
    Matrix* result = create_matrix(m->rows,m->cols);
    int total_elements = m->rows * m -> cols ;
    for (int i = 0; i < total_elements; i++)
    {
       result -> data[i] = m->data[i] * scalar;
    }
    return result;
}