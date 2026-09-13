#ifndef B_ENGINE_H
#define B_ENGINE_H

#include <stdio.h>
#include <time.h>

typedef struct {
    FILE *fp;
    double cumulative_time_ms;
    struct timespec epoch_start;
} BenchLogger;

BenchLogger* logger_init(const char *filename);
void logger_start_epoch(BenchLogger* logger);
void logger_log_epoch(BenchLogger* logger, int epoch, double mse, double mae_sec);
void logger_close(BenchLogger *logger);

#endif