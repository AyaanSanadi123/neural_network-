#ifndef THREADPOOL_H
#define THREADPOOL_H

#include<stdlib.h>
#include<pthread.h>
#include<stdbool.h>

typedef struct {
    void(*execute)(void* arg); // pointer to the function 
    void* arg; // pointer to the argument of that function
} Task;

typedef struct {
    pthread_t* threads;
    int num_threads;

    Task* task_queue;
    int queue_capacity;
    int head;
    int tail;
    int count;
    int pending_tasks;

    pthread_mutex_t lock;
    pthread_cond_t notify;
    pthread_cond_t all_idle;
    bool shutdown;
} ThreadPool;

ThreadPool* thread_pool_init(int num_threads, int queue_capacity);
bool thread_pool_submit(ThreadPool* pool, void (*execute)(void*), void* arg);
void thread_pool_wait(ThreadPool* pool); 
void thread_pool_destroy(ThreadPool* pool);

#endif