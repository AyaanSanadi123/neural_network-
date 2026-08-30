#ifndef LOSSES_H
#define LOSSES_H

#include "matrix.h"
#include <assert.h>

// for regression 
Matrix* mse_derivative(Matrix* predictions, Matrix* expected);

#endif