
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

typedef float real;
typedef unsigned long ulong;
typedef unsigned uint;

//#define STRIP_MINING true
#define ALTERNATIVE_LOOP true
#define PS_OFFSET 14
#define PROBLEM_SIZE (1ULL << PS_OFFSET) // 14
#define BATCH_SIZE 45U
#define RUNS 8U
#define FREE_RUNS 2
#define ALIGNMENT 32U
#define EPS 1e-6F
#define BILLION 1000000000U
#define TILE 8
//#define CHECK_RESULTS

float rand_float()
{
    return ((float) rand() ) / ((float) RAND_MAX); 
}

void aligned_free(void* aligned_ptr) 
{
    int offset = *(((char*)aligned_ptr) - 1);
    free(((char*)aligned_ptr) - offset);
}

void calculate_average(
        real* matrix,
        real* vector_batch,
        real* result_batch,
        real* ref_result_batch,
        ulong N,
        uint runs,
        float gflops_total);


int main()
{
    srand(time(NULL));
    real* matrix = aligned_alloc(
            ALIGNMENT, 
            PROBLEM_SIZE * PROBLEM_SIZE * sizeof(real));

    real* vector_batch = aligned_alloc(
            ALIGNMENT, 
            PROBLEM_SIZE * BATCH_SIZE * sizeof(real));

    real* result_batch = aligned_alloc(
            ALIGNMENT, 
            PROBLEM_SIZE * BATCH_SIZE * sizeof(real));

    real* ref_result_batch = aligned_alloc(
            ALIGNMENT, 
            PROBLEM_SIZE * BATCH_SIZE * sizeof(real));

    const ulong FLOPS = 2ULL * BATCH_SIZE * PROBLEM_SIZE * PROBLEM_SIZE;

    const float GFLOPS = FLOPS * 1.0 / 1000000000;
    
    printf("Initiating benchmark: \
            \n\tProblem Size         : %llu \
            \n\tMatrix Size          : %llu entries \
            \n\tBatch Size           : %llu entries \
            \n\tTotal Data in Matrix : %.2f GiB \
            \n\tTotal Data in Batch  : %.2f MiB\n\n",
            PROBLEM_SIZE,
            PROBLEM_SIZE * PROBLEM_SIZE,
            PROBLEM_SIZE * BATCH_SIZE,
            PROBLEM_SIZE * PROBLEM_SIZE * sizeof(real) * 1.f / (1024 * 1024 * 1024),
            PROBLEM_SIZE * BATCH_SIZE * sizeof(real) * 1.f / (1024 * 1024));


    for (ulong i = 0; i < PROBLEM_SIZE * PROBLEM_SIZE; i++)
    {
        matrix[i] = rand_float();
    }

    for (ulong i = 0; i < PROBLEM_SIZE * BATCH_SIZE; i++)
    {
        vector_batch[i] = rand_float();
    }

    calculate_average(
            matrix,
            vector_batch, 
            result_batch, 
            ref_result_batch, 
            PROBLEM_SIZE, 
            RUNS, 
            GFLOPS);

    //aligned_free(matrix);
    //aligned_free(vector_batch);
    //aligned_free(result_batch);
    //aligned_free(ref_result_batch);
    
    return 0;
}

void batch_mult_opt(
        real* __restrict matrix,
        real* __restrict vector_batch,
        real* __restrict result_batch,
        ulong N)
{
    #ifndef STRIP_MINING
    #pragma omp parallel for schedule(static)
    for (ulong n = 0; n < N; n++)
    {
        for (ulong b = 0; b < BATCH_SIZE; b++)
        {
            real sum = 0.f;

            #pragma omp simd aligned(matrix, vector_batch : ALIGNMENT)
            for (ulong k = 0; k < N; k++)
            {
                sum += matrix[n * N + k] * vector_batch[b * N + k];
            }

            result_batch[b * N + n] = sum;
        }
    }
    #else
    #pragma omp parallel for schedule(static)
    for (ulong n = 0; n < N; n++)
    {
        for (ulong b = 0; b < BATCH_SIZE; b++)
        {

            for (ulong kk = 0; kk + TILE < N; kk += TILE)
            {
                real sum_cache[TILE] = {0.f};
                for (ulong k = kk; k < kk + TILE; k++)
                {
                    sum_cache[k - kk] += matrix[n * N + k] * vector_batch[b * N + k];
                }

                for (ulong k = kk; k < kk + TILE; k++)
                {
                    result_batch[b * N + n] = sum_cache[kk + k];
                }
            }
        }
    }
    #endif // STRIP_MINING
}

#pragma omp declare simd notinbranch aligned(matrix, vector_batch, sums : ALIGNMENT)
inline void dot_product_vectorized(
        real* sums,
        real* matrix,
        real* vector_batch,
        ulong n,
        ulong N,
        ulong k)
{
    for (ulong b = 0; b < BATCH_SIZE; b++) 
    {
        sums[b] += matrix[n * N + k] * vector_batch[b * N + k];
    }
}


void batch_mult_ref(
        real* __restrict matrix,
        real* __restrict vector_batch,
        real* __restrict result_batch,
        ulong N)
{
    for (ulong n = 0; n < N; n++)
    {
        real sums[BATCH_SIZE] = {0.f};
        for (ulong k = 0; k < N; k++) 
        {
            dot_product_vectorized(sums, matrix, vector_batch, n, N, k);
        }

        for (ulong i = 0; i < BATCH_SIZE; i++) 
        {
            result_batch[i * N + n] = sums[i];
        }
    }
}


real check_results(
        real* matrix,
        real* vector_batch,
        real* result_batch,
        real* ref_result_batch,
        ulong N)
{
    batch_mult_ref(matrix, vector_batch, ref_result_batch, N);

    real max_error = 0.f;

    for (ulong i = 0; i < N * BATCH_SIZE; i++)
    {
        real error = fabs(ref_result_batch[i] - result_batch[i]);

        if (error > max_error)
        {
            max_error = error;
        }
    }
    return max_error;
}
real max_value(real* array, ulong size)
{
    ulong i;

    real max_value = array[0];

    for (i = 0; i < size; i++)
    {
        if (array[i] > max_value)
        {
            max_value = array[i];
        }
    }
    return max_value;
}

real min_value(real* array, ulong size)
{
    ulong i;

    real max_value = array[0];

    for (i = 0; i < size; i++)
    {
        if (array[i] < max_value)
        {
            max_value = array[i];
        }
    }
    return max_value;
}

float run(
        real* matrix,
        real* vector_batch,
        real* result_batch,
        real* ref_result_batch,
        ulong N,
        uint run_id)
{
    double seconds;
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    batch_mult_opt(matrix, vector_batch, result_batch, N);

    clock_gettime(CLOCK_MONOTONIC, &end);

    seconds = ((double)end.tv_sec - (double)start.tv_sec) +
        ((double)end.tv_nsec - (double)start.tv_nsec) / 1.0e9;

    #ifdef CHECK_RESULTS
    real diff = check_results(matrix, vector_batch, result_batch, ref_result_batch, N);

    if (diff > EPS)
    {
        real ref_max = max_value(ref_result_batch, N);

        real ref_min = min_value(ref_result_batch, N);

        real refs[2] = {ref_max, ref_min};
        
        real ref_range = max_value(refs, 2);

        real relative_error = diff / (ref_range + 1e-6f) * 100.f;

        if (relative_error > 0.001f)
        {
            printf("\tINCORRECT RESULTS: magnitude \
                    of error: %.5f, relative_error: %.5f\n", 
                    diff, 
                    relative_error);
        }
    }
    #endif // CHECK_RESULTS
    
    printf("Duration: %.5f s. in run %d", seconds, run_id);

    return (float) seconds;
}


void calculate_average(
        real* matrix,
        real* vector_batch,
        real* result_batch,
        real* ref_result_batch,
        ulong N,
        uint runs,
        float gflops_total)
{
    float runtimes[RUNS - FREE_RUNS] = {0.f};
    float gflops_values[RUNS - FREE_RUNS] = {0.f};

    for (uint run_id = 0; run_id < runs; run_id++)
    {
        float runtime = run(matrix, vector_batch, result_batch, ref_result_batch, N, run_id);

        float gflops = gflops_total / runtime; 

        printf(" - GFLOPS: %.5f", gflops);


        if (run_id > FREE_RUNS - 1)
        {
            runtimes[run_id - FREE_RUNS] = runtime;

            gflops_values[run_id - FREE_RUNS] = gflops;

            printf("\n");
        }
        else 
        {
            printf(" - not in average\n");
        }
    }

    float sum_runtime = 0.f;


    float gflop_sum = 0.f;
    
    for (int i = 0; i < runs - FREE_RUNS; i++)
    {
        sum_runtime += runtimes[i];
        gflop_sum += gflops_values[i];
    }

    float avg = sum_runtime / (runs - FREE_RUNS);
    float gflop_avg = gflop_sum / (runs - FREE_RUNS);

    float variance_runtime = 0.f;
    float variance_gflops = 0.f;

    for (int i = 0; i < runs - FREE_RUNS; i++)
    {
        float diff_run = runtimes[i] - avg;
        float diff_gfl = gflops_values[i] - gflop_avg;

        variance_runtime += diff_run * diff_run;
        variance_gflops += diff_gfl * diff_gfl;
        
    }

    variance_runtime /= (runs - FREE_RUNS - 1);
    
    variance_gflops /= (runs - FREE_RUNS - 1);
    
    float stddev_runtime = sqrtf(variance_runtime);
    float stddev_gflops = sqrtf(variance_gflops);


    printf("Average Runtime: %.5f +- %.5f, performance: %.5f +- %.5f GFLOPS\n",
            avg, stddev_runtime, gflop_avg, stddev_gflops);
}
