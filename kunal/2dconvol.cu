#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

#include <cuda_runtime.h>

 
 __global__ void
convolution(const float *A, const float *B, float *C, int a_rows, int a_cols, int b_rows, int b_cols, int c_rows, int c_cols)
{
    int m = blockDim.y * blockIdx.y + threadIdx.y;
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if(m < c_rows && n < c_cols)
	{
		C[m*(c_cols) + n] = 0;
		for(int i=0;i < b_rows;i++)
		{
			for(int j=0; j < b_cols;j++)
			{
				if(((m-i) < a_rows && (m-i) >= 0) && ((n-j) < a_cols && (n-j) >= 0))
				{
					C[m*(c_cols) + n] = B[i*(b_cols) + j] * A[(m-i)*a_cols + (n-j)] + C[m*(c_cols) + n];
				}
			}
		}
	}
	
}

/**
 * Host main routine
 */
int main(int argc, char *argv[])
{
	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
	
	bool first = true;
	float number;
	std::ifstream input(argv[1]);
	int a_rows = 0;
	int a_cols = 0;
	int b_rows = 0;
	int b_cols = 0;
	std::string line;
	
	while(std::getline(input, line))
	{
    if(line.empty())
        first = false;
    std::istringstream element (line);
    if(first == true)
    {
        a_cols = 0;
        while(element >> number)
        {
            a_cols++;
        }
        a_rows++;
    }

    else
    {
        if(!line.empty())
        {
            b_cols = 0;
             while(element >> number)
             {
                 b_cols++;
             }
        b_rows++;
        }
		}
	}
	
	int c_rows = a_rows+b_rows - 1;
	int c_cols = a_cols+b_cols - 1;
	
	size_t size1 = a_rows * a_cols * sizeof(float);
	size_t size2 = b_rows * b_cols * sizeof(float);
	size_t size3 = ((a_rows+b_rows-1)*(a_cols*b_cols-1)) * sizeof(float);
	
	float *h_A = (float *)malloc(size1);
	float *h_B = (float *)malloc(size2);
	float *h_C = (float *)malloc(size3);
	//int numElements = (a_rows+b_rows-1) * (a_cols+b_cols-1);
	
	std::ifstream input1(argv[1]);
	a_rows=a_cols=b_rows=b_cols=0;
	first = true;
	int i=0;
	int j=0;
	
	while(std::getline(input1, line))
	{
    std::istringstream element (line);
    if(line.empty())
        first = false;

    if(first == true)
    {
        a_cols=0;
        while(element >> number)
        {
            h_A[i] = number;
			i++;
            a_cols++;
        }
        a_rows++;
    }

    else
    {
        b_cols = 0;
        if(!line.empty())
        {
            while(element >> number)
            {
                h_B[j] = number;
				j++;
                b_cols++;
            }
            b_rows++;
        }

    }

	}
	
	for(int i = 0; i<a_rows;i++)
	{
		for(int j=0;j<a_cols;j++)
		{
			printf("%.3f ", h_A[i*a_cols+j]);
		}
		printf("\n");
	}
	printf("\n");
	
	for(int i = 0; i<b_rows;i++)
	{
		for(int j=0;j<b_cols;j++)
		{
			printf("%.3f ", h_B[i*b_cols+j]);
		}
		printf("\n");
	}
	
	
	
	
	
	
	printf("%d, %d\n", a_rows, a_cols);
	printf("%d, %d\n", b_rows, b_cols);
	printf("%d, %d\n", c_rows, c_cols);
	
	// Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
	
	
	float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size3);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	// Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size1, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size2, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	// Launch the Vector Add CUDA Kernel
	dim3 dimGrid(((c_rows-1)/2)+1, ((c_cols-1)/2)+1, 1);
	dim3 dimBlock(16, 16, 1);
    //int threadsPerBlock = 256;
    //int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
   // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    convolution<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	
	// Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size3, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	for(int i = 0; i<c_rows; i++)
	{
		for(int j = 0; j < c_cols; j++)
		{
			printf("%.3f ", h_C[i*(c_cols)+j]);
		}
		printf("\n");
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
