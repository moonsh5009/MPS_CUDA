#pragma once

#include <cstdint>
#include <functional>
#include <optional>

#include "cuda_runtime.h"

#define M_PI				3.14159265359

#define MAX_DBLOCKSIZE		2048
#define MAX_BLOCKSIZE		1024
#define DBLOCKSIZE			256
#define BLOCKSIZE			128
#define HBLOCKSIZE			64
#define WARPSIZE			32

#if defined(__CUDACC__)
#	define MCUDA_INLINE_FUNC		__forceinline__
#	define MCUDA_HOST_DEVICE_FUNC	__host__ __device__
#	define MCUDA_HOST_FUNC			__host__
#	define MCUDA_DEVICE_FUNC		__device__
#	define MCUDA_RESTRICT			__restrict__
#else
#	define MCUDA_INLINE_FUNC inline
#	define MCUDA_HOST_DEVICE_FUNC
#	define MCUDA_HOST_FUNC
#	define MCUDA_DEVICE_FUNC
#	define MCUDA_RESTRICT
#endif

#ifndef _DEBUG
#define CUDA_CHECK(x)	(x)
#else
#define CUDA_CHECK(x)	{\
		(x); \
		cudaError_t e = cudaGetLastError(); \
		if (e != cudaSuccess) assert(false); } //"cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
#endif

namespace mcuda
{
	namespace util
	{
		MCUDA_INLINE_FUNC MCUDA_HOST_DEVICE_FUNC uint32_t DivUp(const uint32_t x, const uint32_t y)
		{
			return (y < 1u) ? 1u : (x + y - 1u) / y;
		}

		MCUDA_INLINE_FUNC MCUDA_HOST_DEVICE_FUNC uint32_t Log2(const uint32_t num)
		{
			uint32_t k = 2u, n = 0u;
			while (k << n <= num) n++;
			return n;
		}

		MCUDA_INLINE_FUNC MCUDA_HOST_DEVICE_FUNC uint32_t MaxBinary(const uint32_t num)
		{
			uint32_t n = 1u;
			while (n < num) n = n << 1u;
			return n;
		}

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
	}
}