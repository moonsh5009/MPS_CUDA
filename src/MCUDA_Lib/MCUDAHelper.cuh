#pragma once
#include "MCUDAHelper.h"

#include <cuda/std/type_traits>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace mcuda
{
	namespace util
	{
		template <typename T>
		MCUDA_DEVICE_FUNC constexpr const T& clamp(const T& v, const T& lo, const T& hi)
		{
			return (v < lo) ? lo : (hi < v) ? hi : v;
		}
		template <typename T>
		MCUDA_DEVICE_FUNC constexpr const T& min(const T& a, const T& b)
		{
			return (a < b) ? a : b;
		}
		template <typename T>
		MCUDA_DEVICE_FUNC constexpr const T& max(const T& a, const T& b)
		{
			return (a < b) ? b : a;
		}

		template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC T AtomicMax(T* address, const T val)
		{
			if constexpr (cuda::std::is_same_v<T, double>)
			{
				unsigned long long* address_as_l = reinterpret_cast<unsigned long long*>(address);
				auto old = *address_as_l;
				while (val > __longlong_as_double(old))
				{
					if (const auto assumed = old; (old = atomicCAS(address_as_l, assumed, __double_as_longlong(val))) == assumed)
						break;
				}
				return __longlong_as_double(old);
			}
			else if constexpr (cuda::std::is_same_v<T, float>)
			{
				int* address_as_l = reinterpret_cast<int*>(address);
				auto old = *address_as_l;
				while (val > __int_as_float(old))
				{
					if (const auto assumed = old; (old = atomicCAS(address_as_l, assumed, __float_as_int(val))) == assumed)
						break;
				}
				return __int_as_float(old);
			}
			else
			{
				return atomicMax(address, val);
			}
		}
		template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC T AtomicMin(T* address, const T val)
		{
			if constexpr (cuda::std::is_same_v<T, double>)
			{
				unsigned long long* address_as_l = reinterpret_cast<unsigned long long*>(address);
				auto old = *address_as_l;
				while (val < __longlong_as_double(old))
				{
					if (const auto assumed = old; (old = atomicCAS(address_as_l, assumed, __double_as_longlong(val))) == assumed)
						break;
				}
				return __longlong_as_double(old);
			}
			else if constexpr (cuda::std::is_same_v<T, float>)
			{
				int* address_as_l = reinterpret_cast<int*>(address);
				auto old = *address_as_l;
				while (val < __int_as_float(old))
				{
					if (const auto assumed = old; (old = atomicCAS(address_as_l, assumed, __float_as_int(val))) == assumed)
						break;
				}
				return __int_as_float(old);
			}
			else
			{
				return atomicMin(address, val);
			}
		}
		template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC T AtomicAdd(T* address, const T val)
		{
			if constexpr (cuda::std::is_same_v<T, double>)
			{
				unsigned long long* address_as_l = reinterpret_cast<unsigned long long*>(address);
				unsigned long long old = *address_as_l, assumed;
				do
				{
					assumed = old;
				}
				while ((old = atomicCAS(address_as_l, assumed, __double_as_longlong(val + __longlong_as_double(assumed)))) != assumed);
				return __longlong_as_double(old);
			}
			else if constexpr (cuda::std::is_same_v<T, float>)
			{
				int* address_as_l = reinterpret_cast<int*>(address);
				int old = *address_as_l, assumed;
				do
				{
					assumed = old;
				}
				while ((old = atomicCAS(address_as_l, assumed, __float_as_int(val + __int_as_float(assumed)))) != assumed);
				return __int_as_float(old);
			}
			else
			{
				return atomicAdd(address, val);
			}
		}
		template<typename T>
		MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC T AtomicExch(T* address, const T val)
		{
			if constexpr (cuda::std::is_same_v<T, double>)
			{
				unsigned long long* address_as_l = reinterpret_cast<unsigned long long*>(address);

				unsigned long long old = *address_as_l, assumed;
				do
				{
					assumed = old;
				}
				while ((old = atomicCAS(address_as_l, assumed, __double_as_longlong(val))) != assumed);
				return __longlong_as_double(old);
			}
			else if constexpr (cuda::std::is_same_v<T, float>)
			{
				int* address_as_l = reinterpret_cast<int*>(address);
				int old = *address_as_l, assumed;
				do
				{
					assumed = old;
				}
				while ((old = atomicCAS(address_as_l, assumed, __float_as_int(val))) != assumed);
				return __int_as_float(old);
			}
			else
			{
				return atomicExch(address, val);
			}
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
	}
}