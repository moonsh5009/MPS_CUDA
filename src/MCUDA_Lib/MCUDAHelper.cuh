#pragma once

#include "MCUDAHelper.h"
#include <cuda/std/type_traits>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda.h>

namespace mcuda
{
	namespace util
	{
		template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC T AtomicMax(T* address, const T val)
		{
			if constexpr (cuda::std::is_same_v<T, double>)
			{
				long long* address_as_ull = reinterpret_cast<long long*>(address);
				auto old = *address_as_ull;
				while (val > __longlong_as_double(old))
				{
					if (const auto assumed = old; (old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val))) == assumed)
						break;
				}
				return __longlong_as_double(old);
			}
			else if constexpr (cuda::std::is_same_v<T, float>)
			{
				int* address_as_ull = reinterpret_cast<int*>(address);
				auto old = *address_as_ull;
				while (val > __int_as_float(old))
				{
					if (const auto assumed = old; (old = atomicCAS(address_as_ull, assumed, __float_as_int(val))) == assumed)
						break;
				}
				return __int_as_float(old);
			}
			return atomicMax(address, val);
		}
		template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC T AtomicMin(T* address, const T val)
		{
			if constexpr (cuda::std::is_same_v<T, double>)
			{
				long long* address_as_ull = reinterpret_cast<long long*>(address);
				auto old = *address_as_ull;
				while (val < __longlong_as_double(old))
				{
					if (const auto assumed = old; (old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val))) == assumed)
						break;
				}
				return __longlong_as_double(old);
			}
			else if constexpr (cuda::std::is_same_v<T, float>)
			{
				int* address_as_ull = reinterpret_cast<int*>(address);
				auto old = *address_as_ull;
				while (val < __int_as_float(old))
				{
					if (const auto assumed = old; (old = atomicCAS(address_as_ull, assumed, __float_as_int(val))) == assumed)
						break;
				}
				return __int_as_float(old);
			}
			return atomicMin(address, val);
		}
		/*template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC T AtomicAdd(T* address, const T val)
		{
			if constexpr (cuda::std::is_same_v<T, double>)
			{
				long long* address_as_ull = reinterpret_cast<long long*>(address);
				auto old = *address_as_ull;
				while (const auto assumed = old; (old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)))) != assumed);
				return __longlong_as_double(old);
			}
			else if constexpr (cuda::std::is_same_v<T, float>)
			{
				int* address_as_ull = reinterpret_cast<int*>(address);
				auto old = *address_as_ull;
				while (const auto assumed = old; (old = atomicCAS(address_as_ull, assumed, __float_as_int(val + __int_as_float(assumed)))) != assumed);
				return __int_as_float(old);
			}
			return atomicAdd(address, val);
		}
		template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC T AtomicExch(T* address, const T val)
		{
			if constexpr (cuda::std::is_same_v<T, double>)
			{
				long long* address_as_ull = reinterpret_cast<long long*>(address);
				auto old = *address_as_ull;
				while (const auto assumed = old; (old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val))) != assumed); 
				return __longlong_as_double(old);
			}
			else if constexpr (cuda::std::is_same_v<T, float>)
			{
				int* address_as_ull = reinterpret_cast<int*>(address);
				auto old = *address_as_ull;
				while (const auto assumed = old; (old = atomicCAS(address_as_ull, assumed, __float_as_int(val))) != assumed);
				return __int_as_float(old);
			}
			return atomicAdd(address, val);
		}

		template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC void warpSum(volatile T* sData, const uint32_t tid)
		{
			sData[tid] += sData[tid + 32u];
			sData[tid] += sData[tid + 16u];
			sData[tid] += sData[tid + 8u];
			sData[tid] += sData[tid + 4u];
			sData[tid] += sData[tid + 2u];
			sData[tid] += sData[tid + 1u];
		}
		template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC void warpMin(volatile T* sData, const uint32_t tid)
		{
			if (sData[tid] > sData[tid + 32u])
				sData[tid] = sData[tid + 32u];
			if (sData[tid] > sData[tid + 16u])
				sData[tid] = sData[tid + 16u];
			if (sData[tid] > sData[tid + 8u])
				sData[tid] = sData[tid + 8u];
			if (sData[tid] > sData[tid + 4u])
				sData[tid] = sData[tid + 4u];
			if (sData[tid] > sData[tid + 2u])
				sData[tid] = sData[tid + 2u];
			if (sData[tid] > sData[tid + 1u])
				sData[tid] = sData[tid + 1u];
		}
		template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC void warpMax(volatile T* sData, const uint32_t tid)
		{
			if (sData[tid] < sData[tid + 32u])
				sData[tid] = sData[tid + 32u];
			if (sData[tid] < sData[tid + 16u])
				sData[tid] = sData[tid + 16u];
			if (sData[tid] < sData[tid + 8u])
				sData[tid] = sData[tid + 8u];
			if (sData[tid] < sData[tid + 4u])
				sData[tid] = sData[tid + 4u];
			if (sData[tid] < sData[tid + 2u])
				sData[tid] = sData[tid + 2u];
			if (sData[tid] < sData[tid + 1u])
				sData[tid] = sData[tid + 1u];
		}

		MCUDA_INLINE_FUNC __global__ void ReorderIdsUint2_kernel(
			uint2* xs, uint32_t* ixs, uint32_t size, uint32_t isize)
		{
			extern __shared__ uint32_t s_ids[];
			uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
			uint32_t curr;

			if (id < size)
			{
				uint2 tmp = xs[id];
				curr = tmp.x;
				s_ids[threadIdx.x + 1u] = curr;
				if (id > 0u && threadIdx.x == 0u) {
					tmp = xs[id - 1u];
					s_ids[0] = tmp.x;
				}
			}
			__syncthreads();

			if (id < size) {
				uint32_t i;
				uint32_t prev = s_ids[threadIdx.x];
				if (id == 0u || prev != curr) {
					if (id == 0u) {
						ixs[0] = 0u;
						prev = 0u;
					}
					for (i = prev + 1u; i <= curr; i++)
						ixs[i] = id;
				}
				if (id == size - 1u) {
					for (i = curr + 1u; i < isize; i++)
						ixs[i] = id + 1u;
				}
			}
		}*/
	}
}