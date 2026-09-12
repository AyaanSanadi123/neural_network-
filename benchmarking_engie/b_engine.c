#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct {
    FILE *fp;
    double cumulative_time_ms;
    struct timespec epoch_start;
} BenchLogger;


BenchLogger* logger_init(const char *filename){
    BenchLogger * logger = (BenchLogger*) malloc (sizeof(BenchLogger));
    logger -> fp = fopen(filename,'w');
    if(!logger -> fp){
        perror("Failed to open log file");
        exit(1);
    }
    // set the current time to 0
    logger->cumulative_time_ms = 0.0;
}