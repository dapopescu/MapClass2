/**
 * Copyright 2013 Diana-Andreea Popescu, EPFL, Switzerland.  All rights reserved.
 *
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

#define MAX_EXP	100

inline
void checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
	exit(EXIT_FAILURE);
  }
}

struct is_order_less
{
	__host__ __device__
	bool operator() (const int x)
	{
		return (x < 0);
	}
};

void initPol(unsigned int *exps, unsigned int dim, double *coeffs, unsigned int nvars)
{
    for (unsigned int i = 0; i < dim; ++i)
    {
        for (unsigned int k = 0; k < nvars; ++k)
			exps[i + k * dim] = rand() % 20;
		coeffs[i] = 1;
    }
//	for (unsigned int i = 0; i < dim * nvars; ++i)
//		  printf("%d ", exps[i]);
}

/**
 * Run a multivariate polynomial multiplication using CUDA
 */
int polynomMultiply(int argc, char **argv, int block_size, unsigned int &dimA, unsigned int &dimB, 
		    unsigned int &order, unsigned int &nvars, int n)
{
  //omp_set_num_threads(n);
    // Allocate host memory for polynoms A and B
    unsigned int size_A = dimA * nvars;
    unsigned int mem_size_exp_A = sizeof(unsigned int) * size_A;
    unsigned int mem_size_coeff_A = sizeof(double) * dimA;
    unsigned int *exp_A = (unsigned int*) malloc(mem_size_exp_A);
   
    double *coeff_A = (double*) malloc(mem_size_coeff_A);
   
    unsigned int size_B = dimB * nvars;
    unsigned int mem_size_exp_B = sizeof(unsigned int) * size_B;
    unsigned int mem_size_coeff_B = sizeof(double) * dimB;
    unsigned int *exp_B = (unsigned int*) malloc(mem_size_exp_B);
    double *coeff_B = (double*)malloc(mem_size_coeff_B);

    // Initialize host memory
    initPol(exp_A, dimA, coeff_A, nvars);
    initPol(exp_B, dimB, coeff_B, nvars);

    // Allocate device memory
    double *final_coeff_C;
    unsigned long long *e_keys_C;
    unsigned long long *final_keys_C;

    // Allocate host polynom C
    unsigned int dimC = dimA * dimB;
    unsigned int size_C = dimA * dimB * nvars;
    unsigned int mem_size_exp_C = size_C * sizeof(unsigned int);
    unsigned int *exp_C = (unsigned int*)malloc(mem_size_exp_C);

    unsigned int mem_size_keys_C = dimC * sizeof(unsigned long long); 
    unsigned int mem_size_coeff_C = sizeof(double) * dimC;
    double *coeff_C = (double*)malloc(mem_size_coeff_C);

    final_coeff_C = (double*)malloc(mem_size_coeff_C);

    e_keys_C = (unsigned long long*)malloc(mem_size_keys_C);

    final_keys_C = (unsigned long long*)malloc(mem_size_keys_C);

    //STENCIL FOR TRUNCATION
    int *stencil = NULL;
    unsigned int mem_size_stencil = sizeof(int) * dimC;
    stencil = (int*)malloc(mem_size_stencil);
    
    printf("Computing result ...\n");

    // Execute the kernel
    int nIter = 1;
    unsigned long long ekey = 0, kd = 0;
    unsigned int sum = 0;
    unsigned int cexp = 0;
    for (int it = 0; it < nIter; it++)
    {
      double start1 = omp_get_wtime();
#pragma omp parallel for shared(exp_A, exp_B, exp_C, coeff_A, coeff_B, coeff_C) firstprivate(ekey, cexp, sum) schedule(static)
      for (int i = 0; i < dimB; i ++)
	for (int j = 0; j < dimA; j ++) {
	  coeff_C[i * dimA + j] = coeff_A[j] * coeff_B[i];
	  for (int k = 0; k < nvars; k ++){
	    cexp = exp_A[j + k * dimA] + exp_B[i + k * dimB];
	    exp_C[i * dimA + j + k * dimC] = cexp;
	    ekey = MAX_EXP * ekey + cexp;
	    //sum += cexp;
	  }
	  e_keys_C[i * dimA + j] = ekey;
	
	}
      double end1 = omp_get_wtime();
      printf("%lf\n", 1000 * (end1 - start1));
      	thrust::device_vector<unsigned long long> keys_C_dev(e_keys_C, e_keys_C + dimC);
	thrust::device_vector<double> coeff_C_dev(coeff_C, coeff_C + dimC);
        thrust::device_vector<unsigned long long> final_keys_C_dev(final_keys_C, final_keys_C + dimC);
        thrust::device_vector<double> final_coeff_C_dev(final_coeff_C, final_coeff_C + dimC);
	double start2 = omp_get_wtime();
	thrust::sort_by_key(keys_C_dev.begin(), keys_C_dev.end(), coeff_C_dev.begin());
	thrust::pair<thrust::device_vector<unsigned long long>::iterator, thrust::device_vector<double>::iterator > end;
	end = thrust::reduce_by_key(keys_C_dev.begin(), keys_C_dev.end(), coeff_C_dev.begin(), final_keys_C_dev.begin(), final_coeff_C_dev.begin());
	int sizeC = end.first - final_keys_C_dev.begin();
	double end2 = omp_get_wtime();
	printf("%lf\n", 1000 * (end2 - start2)); 
	double start3 = omp_get_wtime();
#pragma omp parallel for private(kd, ekey) shared(exp_C, final_keys_C_dev, sizeC) schedule(static)
	for (int i = 0; i < sizeC; i ++){
		ekey = final_keys_C_dev[i];
		for (int k = nvars - 1; k >= 0; k--) {
		  	kd = ekey/MAX_EXP;
			exp_C[i + k * dimC] = ekey - kd * MAX_EXP; 
			ekey = kd;
		}
	}
	double end3 = omp_get_wtime();
	printf("%lf\n", 1000 * (end3 - start3));

	
	/*	thrust::device_ptr<int> stencil_dev(stencil);
		thrust::device_ptr<double> end_coeffs_dev = thrust::remove_if(coeffs_C_dev, coeffs_C_dev + dimC, stencil_dev, is_order_less());
		thrust::device_ptr<unsigned long long> end_keys_dev = thrust::remove_if(keys_C_dev, keys_C_dev + dimC, stencil_dev, is_order_less());
		thrust::sort_by_key(keys_C_dev, end_keys_dev, coeffs_C_dev);
		thrust::pair<thrust::device_ptr<unsigned long long>, thrust::device_ptr<double>> end;
		end = thrust::reduce_by_key(keys_C_dev, end_keys_dev, coeffs_C_dev, final_keys_C_dev, final_coeff_C_dev); 
		getExponentsFromKeysCUDA<1024,6,100><<< grid_exp, threads_exp >>>(e_C, final_keys_C, sizeC);
	//	thrust::sort_by_key(keys_C_dev, keys_C_dev + dimC, values_C_dev);
		
	*/
	
    }



    printf("Checking computed result for correctness: ");
    bool correct = true;

/*	for (int i = 0; i < sizeC; ++i){
		for (int k = 0; k < nvars; ++k){
			printf("%d ", exp_C[i + k * sizeC]);
		}
		printf("%lf \n", coeff_C[i]);
	} */
/*    for (unsigned int i = 0; i < dimA; ++i)
    {
    	for (unsigned int j = 0; j < dimB; ++j)
	{
		double coef = coeff_A[i] * coeff_B[j];
	       	for (unsigned int k = 0; k < nvars; ++k)
		{
			unsigned int expc = exp_A[i + k * dimA] + exp_B[j + k * dimB];
			//printf(" %d ", expc);
			if (expc != exp_C[j * dimA + i + k * dimC])
			{
				printf("Error! Pol dif %d - %d %d\n", i, expc, exp_C[j * dimA + i + k * dimC]);
				correct = false;
			}
		}
       	}
	}*/ 

    printf("%s\n", correct ? "OK" : "FAIL");

    // Clean up memory
    free(exp_A);
    free(exp_B);
    free(exp_C);
    free(coeff_A);
    free(coeff_B);
    free(coeff_C);

    free(e_keys_C);
    free(final_keys_C);
    free(final_coeff_C);
    free(stencil);

    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    } 
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Multivariate Polynomial Multiplication Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -nA=NumberOfTermsA (Number of terms of polynom A)\n");
        printf("      -nB=NumberOfTermsB (Number of terms of polynom B)\n");
        printf("	  -x=vars (Number of variables)\n");
		printf("	  -o=order (Order of polynoms).\n");
		printf("	  -b=block_size (Block size).\n");

        exit(EXIT_SUCCESS);
    }

 
    // Use a larger block size for Fermi and above
    int block_size = 16;

    unsigned int dimA = 16 * block_size;
    unsigned int dimB = 8 * block_size;

    // number of terms of polynom A
    if (checkCmdLineFlag(argc, (const char **)argv, "nA"))
    {
        dimA = getCmdLineArgumentInt(argc, (const char **)argv, "nA");
    }

    // number of terms of polynom B
    if (checkCmdLineFlag(argc, (const char **)argv, "nB"))
    {
        dimB = getCmdLineArgumentInt(argc, (const char **)argv, "nB");
    }

	unsigned int order = 6;
	// Order of polynoms
    if (checkCmdLineFlag(argc, (const char **)argv, "o"))
    {
        order = getCmdLineArgumentInt(argc, (const char **)argv, "o");
    }

    int n = 1;
    if (checkCmdLineFlag(argc, (const char **)argv, "n"))
    {
        n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

	unsigned int nvars = 6;
	// Number of variables
    if (checkCmdLineFlag(argc, (const char **)argv, "x"))
    {
		nvars = getCmdLineArgumentInt(argc, (const char **)argv, "x");
    }


    printf("PolynomA(%d), PolynomB(%d), Order = %d, Number of Variables = %d\n", dimA, dimB, order, nvars);

    int polynom_result = polynomMultiply(argc, argv, block_size, dimA, dimB, order, nvars, n);

    exit(polynom_result);
}
