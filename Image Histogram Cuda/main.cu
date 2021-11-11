#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "scrImagePgmPpmPackage.cuh"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

using namespace std;

// calculates the resized image
__global__ void calculateHistogram(unsigned int* imageHistogram, unsigned int width, unsigned int height, cudaTextureObject_t texObj)
{
	const unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	const unsigned int localId = threadIdx.y * blockDim.x + threadIdx.x;
	const unsigned int histStartIndex = (blockIdx.y * gridDim.x + blockIdx.x) * 256;

	__shared__ unsigned int histo_private[256];

	if (localId < 256)
	{
		histo_private[localId] = 0;
	}
	__syncthreads();

	// step 4: read the texture memory from the texture object
	unsigned char imageData = tex2D<unsigned char>(texObj, (float)(tidX), (float)(tidY));
	atomicAdd(&(histo_private[imageData]), 1);

	__syncthreads();

	if (localId < 256)
	{
		imageHistogram[histStartIndex + localId] = histo_private[localId];
	}
}

int main()
{
	int height = 0;
	int width = 0;
	int nHistogram = 0;

	// define the scaling ratio
	unsigned char* data;
	unsigned int* imageHistogram, * d_imageHistogram;

	char inputStr[1024] = { "" };
	cudaError_t returnValue;

	// create a channel description to be used while linking the texture object
	cudaArray* cu_array;
	cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, kind);

	get_PgmPpmParams(inputStr, &height, &width); // get height and width of image
	data = (unsigned char*)malloc(height * width * sizeof(unsigned char));
	printf("Reading image of width (%d) and height (%d)\n", width, height);
	scr_read_pgm(inputStr, data, height, width); // load image data to "data"

	// one histogram per image block. Size range of 0-255 since char image.
	nHistogram = (height / BLOCK_SIZE_Y) * (width / BLOCK_SIZE_X) * 256;
	imageHistogram = (unsigned int*)malloc(nHistogram * sizeof(unsigned int));

	// allocate CUDA array
	returnValue = cudaMallocArray(&cu_array, &channelDesc, width, height);
	if (returnValue != cudaSuccess)
	{
		printf("Error while running CUDA API Malloc Array\n");
	}

	returnValue = cudaMemcpy(cu_array, data, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (returnValue != cudaSuccess)
	{
		printf("Error while running CUDA API Array Copy Host to Device\n");
	}

	// step 1: specify textures
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cu_array;

	// step 2: specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	// step 3: create texture object
	cudaTextureObject_t texObj;
	returnValue = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	
	if (returnValue != cudaSuccess)
	{
		printf("Error while running CUDA API Bind Texture\n");
	}

	returnValue = cudaMalloc(&d_imageHistogram, nHistogram * sizeof(unsigned int));

	if (returnValue != cudaSuccess)
	{
		printf("Error while running CUDA API Malloc\n");
	}

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
	printf("Launching grid with blocks (%d)(%d)", dimGrid.x, dimGrid.y);

	calculateHistogram << <dimGrid, dimBlock >> > (d_imageHistogram, width, height, texObj);

	returnValue = cudaDeviceSynchronize();
	if (returnValue != cudaSuccess)
	{
		printf("Error while running CUDA API kernel\n");
	}

	returnValue = cudaMemcpy(imageHistogram, d_imageHistogram, nHistogram * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (returnValue != cudaSuccess)
	{
		printf("Error while running CUDA API Memcpy Device to Host\n");
	}

	// step 5: destroy texture object
	cudaDestroyTextureObject(texObj);

	printf("Histogram per section is as follows: ");
	for (int i = 0; i < nHistogram / 256; i++)
	{
		printf("---------------------- Histogram for block %d ----------------------\n", i);
		for (int j = 0; j < 256; j++)
		{
			int index = i * 256 + j;
			printf("[%d=[%d]] ", j, imageHistogram[index]);
		}
		printf("\n");
	}

	if (data != NULL)
	{
		free(data);
	}

	if (cu_array != NULL)
	{
		cudaFreeArray(cu_array);
	}

	if (imageHistogram != NULL)
	{
		free(imageHistogram);
	}

	if (d_imageHistogram != NULL)
	{
		cudaFree(d_imageHistogram);
	}

	return 0;
}
