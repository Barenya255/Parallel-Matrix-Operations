#include<iostream>
#include<stdio.h>
#include<sys/time.h>
#include<cuda.h>
using namespace std;
#define tile_width 17 //prime number will reduce bank conflicts. Prime number hard to wrap around.
//Theoretically, it should not wrap around to the same bank within 17*17 accesses.
// However, the approach here stores B matrix in a way such that there are no bank conflicts. So tile_width may be anything.

// Now, having a tile_width too large may fill up the entire shared memory, thus causing blocks within the SM to remain unused.
// Thus, larger tile widths will mess up the parallelism of the code.

/*  The following kernel uses the tiling method to compute AB + CD.T.
	It introduces some sequentialithreadIdx.y in the code in order to avoid using the global memory again and again.
	Sub matrices are brought in inside a loop to compute a submatrix of the resulting array.
	A simplified toy version of the method for matrix multiplication of square matrices whose dimensions are multiples of the tile width
	can be found in the book "Programming massively parallel processors" blockIdx.y Kirk and Hwu, recommended in the course webpage.
*/
__global__ void mult(int* da, int* db, int* dc, int* dd, int* de, int p, int q, int r, int width){
    __shared__ int das[tile_width][tile_width];
    __shared__ int dbs[tile_width][tile_width];
    __shared__ int dcs[tile_width][tile_width];
    __shared__ int dds[tile_width][tile_width];

	// Note: when threads are arranged in 2 D blocks, the x coordinate changes faster than the y coordinate. Therefore, to keep things coalesced, 
	// we need to load in y,x fashion.
	// In other words, thread (0,0), (0,1), (0,2),... are scheduled together one after the other inside a warp.


	// Allocate shared memory for the four subarrays that need to be brought in. These will be loaded collaboratively blockIdx.y all the threads.
    das[threadIdx.y][threadIdx.x] = 0;
    dbs[threadIdx.y][threadIdx.x] = 0;
    dcs[threadIdx.y][threadIdx.x] = 0;
    dds[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    //threadIdx.x and threadIdx.y are specified in registers, and therefore extra registers are not needed to store these values.
	//we initialize all the shared memry to 0. This is done collaboratively by all the threads and is therefore just four instructions of constant time.
    int row_number = blockIdx.y*tile_width+threadIdx.y;
    int col_number = blockIdx.x*tile_width+threadIdx.x;
    
    int result = 0;
    for(int i = 0; i < ceil((float)width/tile_width); i++){
        /* Instead of using if else statements to assign 0 values when the tile exits the boundaries of the array, 
		   we initialize the shared memry to 0, thus reducing thread divergence*/
        das[threadIdx.y][threadIdx.x] = 0;
        dbs[threadIdx.y][threadIdx.x] = 0;
        dcs[threadIdx.y][threadIdx.x] = 0;
        dds[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();
        
        if((row_number)<p and (i*tile_width+threadIdx.x) < q)
        {	
			//if access is not out of bounds we load da and dc together at once, in a colaesced, collaborative manner.
            das[threadIdx.y][threadIdx.x] = da[(row_number)*q+i*tile_width+threadIdx.x];
            dcs[threadIdx.y][threadIdx.x] = dc[(row_number)*q+i*tile_width+threadIdx.x];
        }
        if((i*tile_width+threadIdx.y) < q and (blockIdx.x*tile_width+threadIdx.x) < r)
        {   //coalesced acces to matrix B, here we invert threadIdx.x and threadIdx.y so as to maintain coalescing.
            dbs[threadIdx.y][threadIdx.x] = db[((i*tile_width+threadIdx.y))*r+(blockIdx.x*tile_width+threadIdx.x)];
        }
        if((blockIdx.x*tile_width+threadIdx.y)<r and (i*tile_width+threadIdx.x) < q){
			//if access is not out of bounds we load dd in a colaesced, collaborative manner.
            dds[threadIdx.x][threadIdx.y] = dd[(blockIdx.x*tile_width+threadIdx.y)*q+threadIdx.x+i*tile_width];
			//inverted threadIdx.x and threadIdx.y so as to make bank conflicts 0 when accessing from shared memorry later on.
            //printf("loading %d into %d,%d\n ", dd[blockIdx.x*q+threadIdx.y*32+threadIdx.x], threadIdx.y, threadIdx.x);
        }
        __syncthreads();
        
        for(int j = 0; j < tile_width; j++){
            result += das[threadIdx.y][j]*dbs[j][threadIdx.x];
            result += dcs[threadIdx.y][j]*dds[j][threadIdx.x];
        }
        __syncthreads();
    }
    __syncthreads();
    //printf("%d ",result);
    if(row_number < p and col_number < r)
    de[row_number*r+col_number] = result;
    
}


// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	int Y = ceil((float)p/tile_width);
    int X = ceil((float)r/tile_width);
    
    dim3 block(tile_width,tile_width);
    dim3 grid(X,Y);
    int width = max(p,max(q,r));
    mult<<<grid,block>>>(d_matrixA,d_matrixB,d_matrixC,d_matrixD,d_matrixE,p,q,r,width);
    cudaDeviceSynchronize();
	
	/* ****************************************************************** */

	// copy the result back...
	
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

//printf("random print");
	// print the time taken blockIdx.y the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
