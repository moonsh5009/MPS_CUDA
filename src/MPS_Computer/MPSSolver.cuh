#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "MPSCGKernel.cuh"

#define SOLVER_PRINT		0

namespace mps::kernel
{
	template<typename T>
	uint32_t CGSolver(T* r, T* p, T* Ap, T* x, const T* b, T minError, size_t size, uint32_t maxLoop, std::function<void(T*, T*, uint32_t)> funcComputeAx)
	{
		static_assert(std::is_floating_point<T>::value, "T must be a floating-point type");
		constexpr auto nBlockSize = 256u;
		constexpr auto nMaxBlockSize = 1024u;

		if (size == 0) return 0u;

		thrust::device_vector<T> d_param(2);
		thrust::host_vector<T> h_param;

		funcComputeAx(x, Ap, 0);

		InitCGResidual_kernel << < mcuda::util::DivUp(size, nMaxBlockSize), nMaxBlockSize >> > (
			r, p, b, Ap, size);
		CUDA_CHECK(cudaPeekAtLastError());

		uint32_t l = 1u;
		while (l <= 100u)
		{
			funcComputeAx(p, Ap, l);

			thrust::fill(d_param.begin(), d_param.end(), static_cast<T>(0.0));
			ComputeCGAlphaParam_kernel << < mcuda::util::DivUp(size, nBlockSize << 1), nBlockSize, nBlockSize * 2 * sizeof(T) >> > (
				r, p, Ap, size, thrust::raw_pointer_cast(d_param.data()));
			CUDA_CHECK(cudaPeekAtLastError());

			h_param = d_param;
			if (h_param[0] < minError) break;

		#if SOLVER_PRINT
			std::stringstream ss;
			ss << "CGSolver Error " << l << " : " << h_param[0] << ", " << h_param[1] << std::endl;
			OutputDebugStringA(ss.str().c_str());
		#endif

			const auto alpha = h_param[0] / h_param[1];
			UpdateCGResidual_kernel << < mcuda::util::DivUp(size, nMaxBlockSize), nMaxBlockSize >> > (
				r, x, p, Ap, size, alpha);
			CUDA_CHECK(cudaPeekAtLastError());

			d_param[0] = static_cast<REAL>(0.0);
			ComputeCGBetaParam_kernel << < mcuda::util::DivUp(size, nBlockSize << 1), nBlockSize, nBlockSize * sizeof(T) >> > (
				r, size, thrust::raw_pointer_cast(d_param.data()));
			CUDA_CHECK(cudaPeekAtLastError());

			h_param[1] = d_param[0];
			const auto beta = h_param[1] / h_param[0];
			UpdateCGDirection_kernel << < mcuda::util::DivUp(size, nMaxBlockSize), nMaxBlockSize >> > (
				r, p, size, beta);
			CUDA_CHECK(cudaPeekAtLastError());
			l++;
		}

	#if SOLVER_PRINT
		std::stringstream ss;
		ss << "CGSolver Loop : " << l << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif
		return l;
	}
}