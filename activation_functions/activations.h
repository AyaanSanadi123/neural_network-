#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
    double relu(double x);
    double relu_derivative(double x);
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    double linear(double x);
    double linear_derivative(double x);
    double leaky_relu(double m);
    double leaky_relu_derivative(double m);
#endif