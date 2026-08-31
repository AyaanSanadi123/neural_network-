#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<assert.h>

#include<matrix.h>

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

Matrix * dot_product(Matrix* a,Matrix * b){
    // safelty check 
    assert(a -> cols == b -> rows);
    // create a matrix 
    Matrix * c = create_matrix(a-> rows,b-> cols);
    for(int i = 0;i<a->rows;i++){
        for(int j = 0;j< b -> cols; j++){
            double sum = 0.0;
            for(int k = 0; k < a-> cols; k++){
                // we need to write the formula that converts 2D arrys into a flat 1D array 
                // index = row * col + col
                int a_index = i * a-> cols + k;
                int b_index = k * b -> cols + j;
                sum += a-> data[a_index] * b -> data[b_index];
            }
            c -> data[i * c-> cols + j] = sum;
        }
    }
    return c;
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