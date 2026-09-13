#include<losses.h>
#include<stdio.h>


Matrix* mse_derivative(Matrix* predictions, Matrix* expected){
    assert(predictions->rows == expected->rows && predictions->cols == expected->cols);

    Matrix* dA = create_matrix(predictions->rows, predictions->cols);
    int total_elements = predictions->rows * predictions->cols;
    // divison is a very heavy process, we can just convert it into multiuplication using a math trick
    double batch_size = predictions->cols;
    double inv_batch_size = 1.0 / batch_size;
    for (int i = 0; i < total_elements; i++)
    {
        dA->data[i] = (predictions->data[i] - expected->data[i]) * inv_batch_size;
    }
    return dA;
}