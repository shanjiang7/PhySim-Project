// cuda_kernels.cuh
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <vector>
#include <cstdio>

// 1) add_vector
template<typename T>
__global__ void add_vector_kernel(const T* __restrict__ a,
                                  const T* __restrict__ b,
                                  T* __restrict__ c,
                                  T factor1, T factor2,
                                  int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] * factor1 + b[i] * factor2;
    }
}

template<typename T>
void add_vector(const std::vector<T>& a,
                const std::vector<T>& b,
                std::vector<T>& c,
                T factor1, T factor2,
                int block_size = 256)
{
    int N = a.size();
    c.resize(N);
    T *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(T));
    cudaMalloc(&d_b, N*sizeof(T));
    cudaMalloc(&d_c, N*sizeof(T));
    cudaMemcpy(d_a, a.data(), N*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), N*sizeof(T), cudaMemcpyHostToDevice);

    int grid = (N + block_size - 1) / block_size;
    add_vector_kernel<T><<<grid, block_size>>>(d_a, d_b, d_c, factor1, factor2, N);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, N*sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

// 2) mult_vector
template<typename T>
__global__ void mult_vector_kernel(const T* __restrict__ a,
                                   T* __restrict__ c,
                                   T factor,
                                   int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] * factor;
    }
}

template<typename T>
void mult_vector(const std::vector<T>& a,
                 std::vector<T>& c,
                 T factor,
                 int block_size = 256)
{
    int N = a.size();
    c.resize(N);
    T *d_a, *d_c;
    cudaMalloc(&d_a, N*sizeof(T));
    cudaMalloc(&d_c, N*sizeof(T));
    cudaMemcpy(d_a, a.data(), N*sizeof(T), cudaMemcpyHostToDevice);

    int grid = (N + block_size - 1) / block_size;
    mult_vector_kernel<T><<<grid, block_size>>>(d_a, d_c, factor, N);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, N*sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_c);
}

// 3) add_triplet
template<typename T>
__global__ void add_triplet_kernel_A(const int* __restrict__ rowA,
                                     const int* __restrict__ colA,
                                     const T* __restrict__ valA,
                                     int* __restrict__ rowC,
                                     int* __restrict__ colC,
                                     T* __restrict__ valC,
                                     T factor1,
                                     int Na)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Na) {
        rowC[i] = rowA[i];
        colC[i] = colA[i];
        valC[i] = valA[i] * factor1;
    }
}

template<typename T>
__global__ void add_triplet_kernel_B(const int* __restrict__ rowB,
                                     const int* __restrict__ colB,
                                     const T* __restrict__ valB,
                                     int* __restrict__ rowC,
                                     int* __restrict__ colC,
                                     T* __restrict__ valC,
                                     T factor2,
                                     int Na, int Nb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nb) {
        int idx = Na + i;
        rowC[idx] = rowB[i];
        colC[idx] = colB[i];
        valC[idx] = valB[i] * factor2;
    }
}

template<typename T>
void add_triplet(const std::vector<int>& rowA,
                 const std::vector<int>& colA,
                 const std::vector<T>& valA,
                 const std::vector<int>& rowB,
                 const std::vector<int>& colB,
                 const std::vector<T>& valB,
                 std::vector<int>& rowC,
                 std::vector<int>& colC,
                 std::vector<T>&    valC,
                 T factor1, T factor2,
                 int block_size = 256)
{
    int Na = rowA.size(), Nb = rowB.size();
    int N  = Na + Nb;
    rowC.resize(N);
    colC.resize(N);
    valC.resize(N);

    int *d_rowA, *d_colA, *d_rowC, *d_colC;
    T   *d_valA, *d_valC;
    cudaMalloc(&d_rowA, Na*sizeof(int));
    cudaMalloc(&d_colA, Na*sizeof(int));
    cudaMalloc(&d_valA, Na*sizeof(T));
    cudaMalloc(&d_rowC, N *sizeof(int));
    cudaMalloc(&d_colC, N *sizeof(int));
    cudaMalloc(&d_valC, N *sizeof(T));

    cudaMemcpy(d_rowA, rowA.data(), Na*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colA, colA.data(), Na*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valA, valA.data(), Na*sizeof(T), cudaMemcpyHostToDevice);

    int gridA = (Na + block_size - 1) / block_size;
    add_triplet_kernel_A<T><<<gridA, block_size>>>(d_rowA, d_colA, d_valA,
        d_rowC, d_colC, d_valC, factor1, Na);

    int *d_rowB, *d_colB;
    T   *d_valB;
    cudaMalloc(&d_rowB, Nb*sizeof(int));
    cudaMalloc(&d_colB, Nb*sizeof(int));
    cudaMalloc(&d_valB, Nb*sizeof(T));
    cudaMemcpy(d_rowB, rowB.data(), Nb*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colB, colB.data(), Nb*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valB, valB.data(), Nb*sizeof(T), cudaMemcpyHostToDevice);

    int gridB = (Nb + block_size - 1) / block_size;
    add_triplet_kernel_B<T><<<gridB, block_size>>>(d_rowB, d_colB, d_valB,
        d_rowC, d_colC, d_valC, factor2, Na, Nb);

    cudaDeviceSynchronize();

    cudaMemcpy(rowC.data(), d_rowC, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(colC.data(), d_colC, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(valC.data(), d_valC, N*sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_rowA); cudaFree(d_colA); cudaFree(d_valA);
    cudaFree(d_rowB); cudaFree(d_colB); cudaFree(d_valB);
    cudaFree(d_rowC); cudaFree(d_colC); cudaFree(d_valC);
}

// 4) max_vector (absolute + reduction via Thrust)
template<typename T>
T max_vector(const std::vector<T>& a)
{
    thrust::device_vector<T> d_vec = a;
    // take absolute value in place
    thrust::transform(
        d_vec.begin(), d_vec.end(),
        d_vec.begin(),
        [] __device__ (T x){ return x < T(0) ? -x : x; }
    );
    // find max
    auto iter = thrust::max_element(d_vec.begin(), d_vec.end());
    return iter == d_vec.end() ? T(0) : *iter;
}

// 5) display_vec
template<typename T>
__global__ void display_vec_kernel(const T* __restrict__ v, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        printf("%d %f\n", i, (double)v[i]);
    }
}

template<typename T>
void display_vec(const std::vector<T>& v,
                 int block_size = 256)
{
    T* d_v;
    int N = v.size();
    cudaMalloc(&d_v, N*sizeof(T));
    cudaMemcpy(d_v, v.data(), N*sizeof(T), cudaMemcpyHostToDevice);
    int grid = (N + block_size - 1) / block_size;
    display_vec_kernel<T><<<grid,block_size>>>(d_v, N);
    cudaDeviceSynchronize();
    cudaFree(d_v);
}

// 6) search_dir stub (uses cuSPARSE + cuSOLVER)
//   Solve:  A_csr * dir = -grad  where A_csr is in CSR format.
//   You must set up cusparse/cusolver handles, upload CSR arrays, call
//   cusolverSpScsrlsvqr (or similar), etc.

template<typename T>
void search_dir(const std::vector<T>& grad,      // size N
                const std::vector<int>&   csrRowPtr, // size N+1
                const std::vector<int>&   csrColInd, // size nnz
                const std::vector<T>&     csrVal,    // size nnz
                std::vector<T>& dir)               // size N
{
    int N   = grad.size();
    int nnz = csrVal.size();
    dir.assign(N, T(0));

    // --- Upload inputs to device ---
    thrust::device_vector<T> d_grad = grad;
    thrust::device_vector<int> d_csrRowPtr = csrRowPtr;
    thrust::device_vector<int> d_csrColInd = csrColInd;
    thrust::device_vector<T> d_csrVal    = csrVal;
    thrust::device_vector<T> d_dir(N);

    cusparseHandle_t   cusparseH = nullptr;
    cusolverSpHandle_t cusolverH = nullptr;
    cusparseCreate(&cusparseH);
    cusolverSpCreate(&cusolverH);

    // Build a device CSRDescr, set matrix type etc.
    // Then call cusolverSpScsrlsvqr (for float) or cusolverSpDcsrlsvqr (for double)
    // with parameters: N, nnz, descr, d_csrVal.data().get(),
    //                  d_csrRowPtr.data().get(),
    //                  d_csrColInd.data().get(),
    //                  d_grad.data().get(), tol, reorder, d_dir.data().get(),
    //                  singularity,
    // Then cudaDeviceSynchronize().

    // ... your solver details here ...

    // Copy result back:
    thrust::copy(d_dir.begin(), d_dir.end(), dir.begin());

    cusolverSpDestroy(cusolverH);
    cusparseDestroy(cusparseH);
}