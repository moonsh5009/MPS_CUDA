#pragma once

#include "MPSBaseUtil.cuh"

template<typename T>
__global__ void InitCGResidual_kernel(
	T* MCUDA_RESTRICT r,
	T* MCUDA_RESTRICT p,
	const T* MCUDA_RESTRICT b,
	const T* MCUDA_RESTRICT Ax,
	size_t size)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= size) return;

	const auto residual = b[id] - Ax[id];
	r[id] = residual;
	p[id] = residual;
}

template<typename T>
__global__ void ComputeCGAlphaParam_kernel(
	const T* MCUDA_RESTRICT r,
	const T* MCUDA_RESTRICT p,
	const T* MCUDA_RESTRICT Ap,
	size_t size,
	T* MCUDA_RESTRICT alpha)
{
	extern __shared__ T s_temp[];
	uint32_t id = threadIdx.x + blockIdx.x * (blockDim.x << 1u);

	s_temp[threadIdx.x] = 0.0;
	s_temp[threadIdx.x + blockDim.x] = 0.0;
	if (id < size)
	{
		s_temp[threadIdx.x] = r[id] * r[id];
		s_temp[threadIdx.x + blockDim.x] = p[id] * Ap[id];

		id += blockDim.x;
		if (id < size)
		{
			s_temp[threadIdx.x] += r[id] * r[id];
			s_temp[threadIdx.x + blockDim.x] += p[id] * Ap[id];
		}
	}
#pragma unroll
	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
	{
		__syncthreads();
		if (threadIdx.x < s)
		{
			s_temp[threadIdx.x] += s_temp[threadIdx.x + s];
			s_temp[threadIdx.x + blockDim.x] += s_temp[threadIdx.x + blockDim.x + s];
		}
	}
	__syncthreads();
	if (threadIdx.x < 32u)
	{
		mcuda::util::WarpSum(s_temp, threadIdx.x);
		mcuda::util::WarpSum(s_temp + blockDim.x, threadIdx.x);
		if (threadIdx.x == 0)
		{
			mcuda::util::AtomicAdd(alpha, s_temp[0]);
			mcuda::util::AtomicAdd(alpha + 1, s_temp[blockDim.x]);
		}
	}
}

template<typename T>
__global__ void UpdateCGResidual_kernel(
	T* MCUDA_RESTRICT r,
	T* MCUDA_RESTRICT x,
	const T* MCUDA_RESTRICT p,
	const T* MCUDA_RESTRICT Ap,
	size_t size,
	T alpha)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= size) return;

	x[id] = x[id] + alpha * p[id];
	r[id] = r[id] - alpha * Ap[id];
}

template<typename T>
__global__ void ComputeCGBetaParam_kernel(
	const T* MCUDA_RESTRICT r,
	size_t size,
	T* MCUDA_RESTRICT beta)
{
	extern __shared__ T s_temp[];
	uint32_t id = threadIdx.x + blockIdx.x * (blockDim.x << 1u);

	s_temp[threadIdx.x] = 0.0;
	if (id < size)
	{
		s_temp[threadIdx.x] = r[id] * r[id];

		id += blockDim.x;
		if (id < size)
		{
			s_temp[threadIdx.x] += r[id] * r[id];
		}
	}
#pragma unroll
	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
	{
		__syncthreads();
		if (threadIdx.x < s)
			s_temp[threadIdx.x] += s_temp[threadIdx.x + s];
	}
	__syncthreads();
	if (threadIdx.x < 32u)
	{
		mcuda::util::WarpSum(s_temp, threadIdx.x);
		if (threadIdx.x == 0)
		{
			mcuda::util::AtomicAdd(beta, s_temp[0]);
		}
	}
}

template<typename T>
__global__ void UpdateCGDirection_kernel(
	const T* MCUDA_RESTRICT r,
	T* MCUDA_RESTRICT p,
	size_t size,
	T beta)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= size) return;

	p[id] = r[id] + beta * p[id];
}