#include<math.h>


double relu(double x){
    // if x == 0 retur x, else return x
    if(x == 0.0) return 0.0;
    else return x;
}

double relu_derivative(double x){
    if( x > 0.0) return 1.0;
    else return 0.0;
}

double sigmoid(double x){
    if (x > 6.0) return 1.0;
    if (x < -6.0) return 0.0;
    return (1.0/(1.0 + exp(-x)));
}
double sigmoid_derivative(double x){
    return x * (1.0 - x);
}

double linear(double x){
    return x;
}
double linear_derivative(double x){
    return 1;
}