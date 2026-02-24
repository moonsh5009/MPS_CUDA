#pragma once
#include "MCudaUtil.h"

#include <cuda/std/type_traits>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <glm/glm.hpp>
#include <eigen/Eigen/Dense>

namespace mcuda
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
	MCUDA_DEVICE_FUNC T AtomicMax(T* address, const T val)
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
	MCUDA_DEVICE_FUNC T AtomicMin(T* address, const T val)
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
	MCUDA_DEVICE_FUNC T AtomicAdd(T* address, const T val)
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
	MCUDA_DEVICE_FUNC T AtomicExch(T* address, const T val)
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
	MCUDA_DEVICE_FUNC void WarpSum(T* sData, const uint32_t tid)
	{
		sData[tid] += sData[tid + 32u]; __syncwarp();
		sData[tid] += sData[tid + 16u]; __syncwarp();
		sData[tid] += sData[tid + 8u]; __syncwarp();
		sData[tid] += sData[tid + 4u]; __syncwarp();
		sData[tid] += sData[tid + 2u]; __syncwarp();
		sData[tid] += sData[tid + 1u]; __syncwarp();
	}
	template<typename T>
	MCUDA_DEVICE_FUNC void WarpMin(T* sData, const uint32_t tid)
	{
		if (sData[tid] > sData[tid + 32u]) sData[tid] = sData[tid + 32u]; __syncwarp();
		if (sData[tid] > sData[tid + 16u]) sData[tid] = sData[tid + 16u]; __syncwarp();
		if (sData[tid] > sData[tid + 8u]) sData[tid] = sData[tid + 8u]; __syncwarp();
		if (sData[tid] > sData[tid + 4u]) sData[tid] = sData[tid + 4u]; __syncwarp();
		if (sData[tid] > sData[tid + 2u]) sData[tid] = sData[tid + 2u]; __syncwarp();
		if (sData[tid] > sData[tid + 1u]) sData[tid] = sData[tid + 1u]; __syncwarp();
	}
	template<typename T>
	MCUDA_DEVICE_FUNC void WarpMax(T* sData, const uint32_t tid)
	{
		if (sData[tid] < sData[tid + 32u]) sData[tid] = sData[tid + 32u]; __syncwarp();
		if (sData[tid] < sData[tid + 16u]) sData[tid] = sData[tid + 16u]; __syncwarp();
		if (sData[tid] < sData[tid + 8u]) sData[tid] = sData[tid + 8u]; __syncwarp();
		if (sData[tid] < sData[tid + 4u]) sData[tid] = sData[tid + 4u]; __syncwarp();
		if (sData[tid] < sData[tid + 2u]) sData[tid] = sData[tid + 2u]; __syncwarp();
		if (sData[tid] < sData[tid + 1u]) sData[tid] = sData[tid + 1u]; __syncwarp();
	}

	template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD, class Func>
	MCUDA_DEVICE_FUNC void kernel_Loop(size_t size, Func&& func)
	{
		constexpr auto BLOCK_STRIDE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

		const auto baseThreadId = blockIdx.x * BLOCK_STRIDE + threadIdx.x;

	#pragma unroll
		for (size_t bid = 0; bid < BLOCK_STRIDE; bid += BLOCK_SIZE)
		{
			const auto tid = baseThreadId + bid;
			if (tid < size)
			{
				func(tid);
			}
		}
	}

	template<class Func>
	MCUDA_DEVICE_FUNC void kernel_LoopCG(size_t elementsPerThread, size_t size, Func&& func)
	{
		const auto BLOCK_STRIDE = blockDim.x * elementsPerThread;

		const auto baseThreadId = blockIdx.x * BLOCK_STRIDE + threadIdx.x;

	#pragma unroll
		for (size_t bid = 0; bid < BLOCK_STRIDE; bid += blockDim.x)
		{
			const auto tid = baseThreadId + bid;
			if (tid < size)
			{
				func(tid);
			}
		}
	}

	inline int GetMaxCooperativeBlocks(int device, void* kernelFunc, int blockSize)
	{
		int numBlocksPerSm = 0;
		int numSMs = 0;

		// 1. SM당 최대 블록 수 계산
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&numBlocksPerSm,
			kernelFunc,
			blockSize,
			0));

		// 2. 디바이스의 SM 개수 확인
		cudaDeviceProp deviceProp;
		CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));
		numSMs = deviceProp.multiProcessorCount;

		// 3. 전체 최대 블록 수
		int maxBlocks = numBlocksPerSm * numSMs;

		/*Logger::Info("=== Cooperative Kernel Info ===");
		Logger::Info("Device: ", deviceProp.name);
		Logger::Info("Compute Capability: ", deviceProp.major, ".", deviceProp.minor);
		Logger::Info("SM Count: ", numSMs);
		Logger::Info("Max Blocks Per SM: ", numBlocksPerSm);
		Logger::Info("Max Total Blocks: ", maxBlocks);
		Logger::Info("Block Size: ", blockSize);
		Logger::Info("Max Threads: ", maxBlocks * blockSize);
		Logger::Info("==============================");
		Logger::Print();*/

		return maxBlocks;
	}
	
	template<typename T>
	MCUDA_HOST_DEVICE_FUNC Eigen::Matrix<T, 3, 1> Convert(const glm::vec<3, T>& x)
	{
		return { x.x, x.y, x.z };
	}
	template<typename T>
	MCUDA_HOST_DEVICE_FUNC glm::vec<3, T> Convert(const Eigen::Matrix<T, 3, 1>& x)
	{
		return { x.x(), x.y(), x.z() };
	}
}
