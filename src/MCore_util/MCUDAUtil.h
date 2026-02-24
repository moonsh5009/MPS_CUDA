#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>

#ifdef __CUDACC__
#include <nvtx3/nvToolsExt.h>
#endif

#ifdef MCORE_USE_CUDA
#include <cuda_runtime.h>
#endif

#define M_PI				3.14159265359

#define MAX_DBLOCKSIZE		2048
#define MAX_BLOCKSIZE		1024
#define DBLOCKSIZE			256
#define BLOCKSIZE			128
#define HBLOCKSIZE			64
#define WARPSIZE			32

#ifdef __CUDA_RUNTIME_H__
#	define MCUDA_HOST_DEVICE_FUNC	__host__ __device__
#	define MCUDA_HOST_FUNC			__host__
#	define MCUDA_DEVICE_FUNC		__device__
#	define MCUDA_RESTRICT			__restrict__
#	define MCUDA_FORCE_INLINE		__forceinline__
#else
#	define MCUDA_HOST_DEVICE_FUNC
#	define MCUDA_HOST_FUNC
#	define MCUDA_DEVICE_FUNC
#	define MCUDA_RESTRICT
#	define MCUDA_FORCE_INLINE		inline
#endif

#ifdef __CUDA_RUNTIME_H__
#ifndef _DEBUG
#define CUDA_CHECK(x)	(x)
#else
#define CUDA_CHECK(x) \
{ \
	(x); \
	cudaError_t e = cudaDeviceSynchronize(); \
	if (e != cudaSuccess) \
	{ \
		std::stringstream ss; \
		ss << "cuda failure " << __FILE__ << ":" << __LINE__ << ":" << cudaGetErrorString(e) << "\n"; \
		OutputDebugStringA(ss.str().c_str()); \
		assert(false); \
	} \
	e = cudaGetLastError(); \
	if (e != cudaSuccess) \
	{ \
		std::stringstream ss; \
		ss << "cuda failure " << __FILE__ << ":" << __LINE__ << ":" << cudaGetErrorString(e) << "\n"; \
		OutputDebugStringA(ss.str().c_str()); \
		assert(false); \
	} \
}
#endif
#else
#define CUDA_CHECK(x)	(x)
#endif

namespace mcore
{
	template<size_t BLOCK_SIZE>
	MCUDA_HOST_DEVICE_FUNC constexpr uint32_t DivUp(const uint32_t x)
	{
		return (x + BLOCK_SIZE - 1u) / BLOCK_SIZE;
	}

	template<size_t BLOCK_SIZE>
	MCUDA_HOST_DEVICE_FUNC constexpr uint32_t DivUp(const size_t x)
	{
		return static_cast<uint32_t>((x + BLOCK_SIZE - 1u) / BLOCK_SIZE);
	}

	MCUDA_HOST_DEVICE_FUNC constexpr uint32_t DivUp(const uint32_t x, const uint32_t y)
	{
		return (x + y - 1u) / y;
	}

	MCUDA_HOST_DEVICE_FUNC constexpr uint32_t Log2(const uint32_t num)
	{
		uint32_t k = 2u, n = 0u;
		while (k << n <= num) n++;
		return n;
	}

	MCUDA_HOST_DEVICE_FUNC constexpr uint32_t MaxBinary(const uint32_t num)
	{
		uint32_t n = 1u;
		while (n < num) n = n << 1u;
		return n;
	}

#ifdef __CUDA_RUNTIME_H__
	struct SortUint2CMP
	{
		MCUDA_HOST_DEVICE_FUNC constexpr bool operator()(const uint2& a, const uint2& b) const
		{
			if (a.x != b.x)
				return a.x < b.x;
			return a.y < b.y;
		}
	};

	struct TransformUint2CMP
	{
		MCUDA_HOST_DEVICE_FUNC constexpr uint32_t operator()(const uint2& a) const
		{
			return a.y;
		}
	};
#endif
}