#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "b_engine.h"


BenchLogger* logger_init(const char *filename){
    BenchLogger * logger = (BenchLogger*) malloc (sizeof(BenchLogger));
    logger -> fp = fopen(filename,"w");
    if(!logger -> fp){
        perror("Failed to open log file");
        exit(1);
    }
    // set the current time to 0
    logger->cumulative_time_ms = 0.0;

    fprintf(logger->fp, "epoch,epoch_time_ms,total_time_ms,rmse_ms,mae_sec\n");
    fflush(logger->fp);
    return logger;
}

void logger_start_epoch(BenchLogger* logger){
    // asks the OS's internal clock for 
    clock_gettime(CLOCK_MONOTONIC,&logger -> epoch_start);
}

// record the metrices and calculate the running totals 

void logger_log_epoch(BenchLogger* logger,int epoch,double mse,double mae_sec){
    struct timespec epoch_end;

    clock_gettime(CLOCK_MONOTONIC,&epoch_end);
    // get the epoch time in ms
    double epoch_time_ms = (double)(epoch_end.tv_sec - logger->epoch_start.tv_sec) * 1000.0 +
                           (double)(epoch_end.tv_nsec - logger->epoch_start.tv_nsec) / 1000000.0;

    
    logger -> cumulative_time_ms += epoch_time_ms;
    double rmse_ms = sqrt(mse);


    fprintf(logger->fp, "%d,%.3f,%.3f,%.2f,%.4f\n",
            epoch, epoch_time_ms, logger->cumulative_time_ms, rmse_ms, mae_sec);

    fflush(logger -> fp);

    printf("[Epoch %3d] Time: %8.2f ms | Total: %8.2f ms | MAE: %6.3f s \n",
           epoch, epoch_time_ms, logger->cumulative_time_ms, mae_sec);
}

void logger_close(BenchLogger *logger) {
    if (logger->fp) {
        fclose(logger->fp);
    }
    free(logger);
}