/**
 * Name : Veerakumar Natarajan
 * Student Id: 200208042
 *
 * 2d convolution program
 */

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the 2d convolution of A and B into C.
 */
__global__ void
convol(float *A, float *B, float *C, int row_a, int row_b, int row_c, int col_a, int col_b, int col_c)
{
    int x = blockDim.y * blockIdx.y + threadIdx.y;
	int y = blockDim.x * blockIdx.x + threadIdx.x;
	C[x * col_c + y] = 0.0;
	if(x < row_c && y < col_c) {
	for(int i = 0; i < row_b; i++) {
		for(int j = 0; j < col_b; j++) {
			if(((x - i) < row_a && (x - i) >= 0) && ((y - j) < col_a && (y - j) >= 0))
				C[x * col_c + y] += B[i * col_b + j] * A[(x - i) * col_a + (y - j)];
		}
	}
	}
}

/**
 * Host main routine
 */
int
main(int argc, char *argv[])
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

	
	float *h_A, *h_B, *h_C, tmp;
	int row_a, row_b, row_c, col_a, col_b, col_c;
	int a_matrix = 1; 
	int i, j;
	int size_a, size_b, size_c;
	std::ifstream file(argv[1]);
	std::string row;		
	row_a=row_b=row_c=col_a=col_b=col_c=0;

	// Finding size of matrix A and matrix B 
	while(std::getline(file, row)) {
		if(row.empty())
			a_matrix = 0;

		std::istringstream iss(row);
		if(a_matrix == 1) {
			col_a=0;
			while(iss.good()) {
				iss >> tmp;
				col_a++;
			}
			row_a++;
		} else {
			if(!row.empty()) {
				col_b=0;
				while(iss.good()) {
					iss >> tmp;
					col_b++;
				}
				row_b++;
			}
		}
	}

	row_c = row_a + row_b - 1;
	col_c = col_a + col_b - 1;

	// Calculating size of matrix A, B and C
	size_a = row_a * col_a;
	size_b = row_b * col_b;
	size_c = row_c * col_c;

    // Allocate the host input vector A, B
    h_A = (float *)malloc(size_a * sizeof(float));

    h_B = (float *)malloc(size_b * sizeof(float));

	// Allocate the host output vector
    h_C = (float *)malloc(size_c * sizeof(float));

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

	// Reading value of matrix A and B from input file
	std::ifstream file1(argv[1]);
	a_matrix = 1;
	i = j = 0;
	while(std::getline(file1, row)) {
		if(row.empty())
			a_matrix = 0;
		std::istringstream iss1(row);
		 if(a_matrix == 1){
			while(iss1.good()) {
				iss1 >> tmp;
				h_A[i] = tmp;
				i++;
			}
		} else {
			if(!row.empty()) {
				while(iss1.good()) {
					iss1 >> tmp;
					h_B[j] = tmp;
					j++;
				}
			}
		}
	}

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size_a * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size_b * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size_c * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    err = cudaMemcpy(d_A, h_A, size_a * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size_b * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector 2dconvol CUDA Kernel
	dim3 dimBlock(row_c, col_c, 1);
	dim3 dimGrid(4, 4, 1);
 
	convol<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, row_a, row_b, row_c, col_a, col_b, col_c);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(h_C, d_C, size_c * sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	for(i = 0; i < row_c; i++) {
		for(j = 0; j < col_c; j++) {
			printf("%.3f ", h_C[i * col_c + j]);
		}
		printf("\n");
	}
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

