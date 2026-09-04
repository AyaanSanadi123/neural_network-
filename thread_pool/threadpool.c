#include <stdlib.h>
#include <stdio.h>
#include<stdbool.h>
#include "threadpool.h"


static void* worker_loop(void* arg);


ThreadPool* thread_pool_init(int num_threads,int queue_capacity){
    if(num_threads <= 0 || queue_capacity <= 0) {
        printf("Please enter a valid number of threads or queue size\n");
        return NULL;
    }

    ThreadPool* pool = (ThreadPool*) malloc(sizeof(ThreadPool));

    if (pool == NULL)
    {
        printf("Could not allocate Pool\n");
        return NULL;
    }

    pool -> num_threads = num_threads;
    pool -> queue_capacity = queue_capacity;
    pool -> head = 0;
    pool -> tail = 0;
    pool -> count = 0;
    pool -> pending_tasks = 0;
    pool -> shutdown = false;


    pool -> threads = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
    pool -> task_queue = (Task*) malloc(queue_capacity * sizeof(Task));


    if (pool -> threads == NULL || pool -> task_queue == NULL)
    {
       printf("Cound not allocate memory for threads or task queue\n");
       free(pool->threads);
       free(pool -> task_queue);
       free(pool);

       return NULL;
    }

    if ( pthread_mutex_init(&(pool->lock),NULL) != 0 ||
         pthread_cond_init(&(pool->notify),NULL) != 0 ||
         pthread_cond_init(&(pool -> all_idle),NULL) != 0
        )
    {
       printf("Cound not allocate memory to mutex or condition variables\n");
       free(pool->threads);
       free(pool -> task_queue);
       free(pool);
       return NULL;
    }
    
    for (int i = 0; i < num_threads; i++)
    {
        if(pthread_create(&(pool->threads[i]),NULL,worker_loop,(void*)pool) != 0){

            // if a thread fails to create, we need to make sure all the created threads are 
            // deleted carefully and then destroy the pool

            pthread_mutex_lock(&(pool->lock));
            pool -> shutdown = true;
            pthread_cond_broadcast(&(pool -> notify)); // wakeup all the threads 
            pthread_mutex_unlock(&(pool -> lock));

            // wait for all the created threads to finish their execuation
            for (int j = 0; j < i; j++) {
                pthread_join(pool->threads[j], NULL);
            }

            // now destroy the pool 
            pthread_mutex_destroy(&(pool->lock));
            pthread_cond_destroy(&(pool->notify));
            free(pool->threads);
            free(pool->task_queue);
            free(pool);
            
            return NULL;
        }
    }
    
    return pool;
}

// think of this as the consumer, it always eats the tasks from the queue 
static void* worker_loop(void* arg){
    // caste the argument into a pool obj 
    ThreadPool * pool = (ThreadPool*)arg;

    while(1){
        pthread_mutex_lock(&(pool->lock));
        // if the queue is empty and does not need to shutdown
        while(pool -> count == 0 && !pool->shutdown){
            pthread_cond_wait(&(pool->notify),&(pool->lock));
        }

        // if shutdown is true and the queue is empty, break out of the while(1)
        if(pool -> shutdown && pool -> count == 0){
            pthread_mutex_unlock(&(pool -> lock));
            break;
        }

        // at this point, one thread gets the task

        // pop the task from the ring buffer 
        Task task = pool -> task_queue[pool -> head];

        // advance the head 
        pool -> head = (pool -> head + 1) % pool -> queue_capacity;
        pool -> count --;
        // unlock, so other threads can pick up tasks as the current one is being executed 
        pthread_mutex_unlock(&(pool -> lock));

        // execute the task 
        if(task.execute != NULL) task.execute(task.arg);

        // when the task finishes 
        pthread_mutex_lock(&(pool -> lock));
        pool -> pending_tasks --;

        // if this was the last task, then all threads are idle. 
        // if all threads are idle, wakeup the main thread to update the matrix 
        if(pool -> pending_tasks == 0){
            pthread_cond_signal(&(pool -> all_idle));
        }
        pthread_mutex_unlock(&(pool -> lock));
    }
    return NULL;
}

// think of this as the producer, it fills the queue with tasks 
bool thread_pool_submit(ThreadPool* pool, void (*execute)(void*), void* arg){
    if (pool == NULL || execute == NULL) return false;

    pthread_mutex_lock(&(pool -> lock));

    // check for overflow,
    // if we have an overflow reject the task
    // the rest can be managed in the main loop 
     if (pool -> count == pool -> queue_capacity){
        pthread_mutex_unlock(&(pool -> lock));
        return false;
     }
     
     // do not accept new tasks if shutdown has been initiated 
     if (pool -> shutdown)
     {
        pthread_mutex_unlock(&(pool -> lock));
        return false;
     }

     // now we package the task into the ring queue 
     pool -> task_queue[pool -> tail].execute = execute;
     pool -> task_queue[pool -> tail].arg = arg;
     
     pool -> tail = (pool->tail + 1) % pool->queue_capacity;

     pool -> count ++;
     pool -> pending_tasks ++;

     pthread_cond_signal(&(pool -> notify));
     pthread_mutex_unlock(&(pool -> lock));
     return true;
     
}

void thread_pool_wait(ThreadPool* pool) {
    if (pool == NULL) return;

    pthread_mutex_lock(&(pool->lock));
    
    while (pool->pending_tasks > 0) {
        pthread_cond_wait(&(pool->all_idle), &(pool->lock));
    }
    
    pthread_mutex_unlock(&(pool->lock));
}

void thread_pool_destroy(ThreadPool* pool){
    if(pool == NULL) return;

    // begin the shutdown process 
     pthread_mutex_lock(&(pool -> lock));

     pool -> shutdown = true ;

     // send a sigal to all the threads, wake them up
     pthread_cond_broadcast(&(pool -> notify));
     // unlock and let the threads finish execution,
     // if you dont unlock now, you create a deadlock 
     pthread_mutex_unlock(&(pool -> lock));

     // wait for all the threads to finish execution
     for (int i = 0; i < pool -> num_threads; i++)
     {
       if (pthread_join(pool->threads[i], NULL) != 0) {
            fprintf(stderr, "Warning: Failed to join thread %d during teardown\n", i);
        }
     }
     // destroy the mutex,cond
    pthread_mutex_destroy(&(pool->lock));
    pthread_cond_destroy(&(pool->notify));
    pthread_cond_destroy(&(pool->all_idle));

    // free the pool 
    free(pool->threads);
    free(pool->task_queue);
    free(pool);
}