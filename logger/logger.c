#include "logger.h"
#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <math.h>


// this is a logger, this module's soul reponsibility is to take dynamic input's and store them in a csv
// it has no idea weather we are logging a regression/classification/etc, just take inputs and store them in a csv as output 



BenchLogger* logger_init(const char *filename,const char** metric_names,int num_metrics);
void logger_start_epoch(BenchLogger* logger);
void logger_log_epoch(BenchLogger* logger,int epoch,double* metric_values);
