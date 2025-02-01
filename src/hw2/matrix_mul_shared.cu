#include <stdio.h>

// these are just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

const int DSIZE = 8192;
const int block_size = 32; // CUDA maximum is 1024 *total* threads in block
const float A_val = 3.0f;
const float B_val = 2.0f;

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int idx = threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
  // printf("blockdim.x = %d\n", blockDim.x);
  int idy = threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds)) {
    float temp = 0;
    for (int i = 0; i < ds / block_size; i++) { // 块操作？ 不是，cuda kernel对应的是线程函数，所有的线程都会做。 
    // 这一步是将矩阵乘法拆分为求和操作，每一行每一列相乘再求和。分批操作， 每一个线程块的目的是为了求对应的C矩阵的一个元素。
    // grid分割 (blockIdx.y, blockIdx.x)
    //  grid = [ (0, 0) | (0, 1) | (0, 2) ]
    //         [--------|--------|--------]
    //         [ (1, 0) | (1, 1) | (1, 2) ]
    //         [--------|--------|--------]
    //         [ (2, 0) | (2, 1) | (2, 2) ]  
    // 线程块的分割 (threadIdx.y, threadIdx.x)
    //  thread block = [ (0, 0) | (0, 1) ]
    //                 [--------|--------]
    //                 [ (1, 0) | (1, 1) ]
    //                 [--------|--------]
    // 每个小块的大小为 block_size * block_size
    // 比如 原矩阵A[6,6]维，设置block_size为2，经过分块后，分割为九个二维矩阵，A00为[2,2]矩阵, 线程块也是2维矩阵
    // 每个小块的行或列偏移为 i * block_size (i = 0,1,2, ..., ds/block_size)
    // A = [ A00 | A01 | A02 ]
    //     [-----|-----|-----]
    //     [ A10 | A11 | A12 ]
    //     [-----|-----|-----]
    //     [ A20 | A21 | A22 ]
    // 目前位于 idy 行， 每一个块内的行元素为(i * block_size + threadIdx.x)。
    // A是按行存储的，所以一维向量索引为 idy * ds + (i*block_size + threadIdx.x) 
    // 
    // 同理, B也分割为九个二维矩阵， B00为[2,2]矩阵
    // B = [ B00 | B01 | B02 ]
    //     [-----|-----|-----]
    //     [ B10 | B11 | B12 ]
    //     [-----|-----|-----]
    //     [ B20 | B21 | B22 ]
    // 目前位于 idx 列， 每一块的起点位于 (i*block_size + threadIdx.y)*ds 行， 找到行数后对应的列数是固定的，即idx列(二维)。
    // B也是按行存储的，所以一维向量索引为 (i * block_size + threadIdx.y) * ds + idx
    // 针对线程块，他是一整个在提取，然后共享内存，所以需要sync thread, 获取的是一块内容，不是单列， threadIdx.y, idx都是相对固定值 
    //
    // 针对线程(0,0) ,它的目标是求取C00 = A00 * B00 + A01 * B10 + A02 * B20
    // i = 0时，As = A00, Bs = B00
    // i = 1时，As = A01, Bs = B10
    // i = 0时，As = A02, Bs = B20


    // 当前位置的C对应的index是idy*ds+idx， 存储的As与Bs在共享内存中，在块内共享，块外不共享。
    // 取第idy行的A 与 第idx列的B 
    // 如果要获取当前index的C值，需要索引出当前位置的A与B矩阵，
    // A与B都是按行赋值
    // A对应的是取行向量，目前位于idy行， idy*ds得到目前A的索引起点 ，根据threadIdx.x分配列值，最后因为矩阵有分块偏移，需要全部索引后做加法，索引需要加上i * block_size
      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)]; 
      
    // B对应的是取列向量，目前位于idx列，索引的目标是把列分块导出，那么此时的索引行起点为 (i * block_size + threadIdx.y) * ds， 这样每一个线程对应的起点都是从0块开始，
    // 再根据idx从起点分配列值，得到第idx列的B
      Bs[threadIdx.y][threadIdx.x] = B[idx + (i * block_size + threadIdx.y) * ds]; 
      
      // Synchronize，所有线程会在这里同步完，都进行赋值操作才会进下一步
      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < block_size; k++)
        temp += As[threadIdx.y][k] *
                Bs[k][threadIdx.x]; // dot product of row and column
      __syncthreads();
    }

    // Write to global memory
    C[idy * ds + idx] = temp;
  }
}

int main() {

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;

  // start timing
  t0 = clock();

  h_A = new float[DSIZE * DSIZE];
  h_B = new float[DSIZE * DSIZE];
  h_C = new float[DSIZE * DSIZE];
  for (int i = 0; i < DSIZE * DSIZE; i++) {
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;
  }

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
  cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
  cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size); // dim3 variable holds 3 dimensions
  dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < DSIZE * DSIZE; i++)
    if (h_C[i] != A_val * B_val * DSIZE) {
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i],
             A_val * B_val * DSIZE);
      return -1;
    }
  printf("Success!\n");
  return 0;
}
