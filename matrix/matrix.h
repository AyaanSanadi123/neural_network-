#ifndef MATRIX_H
#define MATRIX_H
typedef struct{
    int cols;
    int rows;
    double* data ;
} Matrix;
Matrix * create_matrix(int rows,int cols );
void free_matrix(Matrix * m);
Matrix * dot_product(Matrix* a,Matrix * b);
Matrix* transpose(Matrix * m);
Matrix * hadamard_product(Matrix* a,Matrix* b);
Matrix* matrix_copy(Matrix* m);
Matrix* matrix_add(Matrix* a, Matrix* b);
Matrix* matrix_map(Matrix* m, double (*func)(double));

#endif