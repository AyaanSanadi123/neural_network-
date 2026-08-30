#include<losses.h>
#include<stdio.h>


Matrix* mse_derivative(Matrix* predictions, Matrix* expected){
    assert(predictions->rows == expected->rows && predictions->cols == expected->cols);

    Matrix* dA = create_matrix(predictions->rows, predictions->cols);
    int total_elements = predictions->rows * predictions->cols;

    for (int i = 0; i < total_elements; i++)
    {
        dA->data[i] = predictions->data[i] - expected->data[i];
    }
    return dA;
}