#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <time.h>

typedef struct {
    FILE *fp;
    double cumulative_time_ms;
    struct timespec epLOGGER_H
} BenchLogger;

BenchLogger* logger_init(const char *filename,const char** metric_names,int num_metrics);
void logger_start_epoch(BenchLogger* logger);
void logger_log_epoch(BenchLogger* logger,int epoch,double* metric_values);

#endif