#include "stdafx.h"
#include "SPHUtil.cuh"
#include <thrust/extrema.h>
#include "thrust/host_vector.h"
#include "AdvectUtil.h"

#define DEBUG_PRINT		0

namespace
{
	constexpr auto nBlockSize = 256u;
}

void mps::kernel::sph::ComputeDensity(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeDensity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ComputeDFSPHFactor(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeDFSPHFactor_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ComputeDFSPHConstantDensity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL> d_error(1);
	thrust::host_vector<REAL> h_error(1);

	uint32_t l = 0u;
	while (l < 100u)
	{
		d_error.front() = static_cast<REAL>(0.0);

		ComputeCDStiffness_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_error.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_error = d_error;
		h_error.front() /= static_cast<REAL>(sph.GetSize() + DBL_EPSILON);
		if (h_error.front() < 1.0e-4 && l >= 2u) break;

	#if DEBUG_PRINT
		std::stringstream ss;
		ss << "Pressure Error " << l << " : " << h_error.front() << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif

		mps::kernel::ResetForce(sph);
		ApplyDFSPH_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(sph, material, hash);
		CUDA_CHECK(cudaPeekAtLastError());
		mps::kernel::UpdateVelocity(physParam, sph);
		l++;
	}
}

void mps::kernel::sph::ComputeDFSPHDivergenceFree(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL> d_error(1);
	thrust::host_vector<REAL> h_error(1);

	uint32_t l = 0u;
	while (l < 100u)
	{
		d_error.front() = static_cast<REAL>(0.0);

		ComputeDFStiffness_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_error.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_error = d_error;
		h_error.front() /= static_cast<REAL>(sph.GetSize() + DBL_EPSILON);
		if (h_error.front() < 1.0e-3 && l >= 1u) break;

	#if DEBUG_PRINT
		std::stringstream ss;
		ss << "DivergenceFree Error " << l << " : " << h_error.front() << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif

		mps::kernel::ResetForce(sph);
		ApplyDFSPH_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(sph, material, hash);
		CUDA_CHECK(cudaPeekAtLastError());
		mps::kernel::UpdateVelocity(physParam, sph);
		l++;
	}
}

void mps::kernel::sph::ApplyExplicitViscosity(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ApplyExplicitViscosity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ApplyExplicitSurfaceTension(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ApplyExplicitSurfaceTension_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ApplyImplicitViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL> d_error(1);
	thrust::host_vector<REAL> h_error(1);

	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPredictVel()));
	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPreviousVel()));

	REAL omega = 1.0;
	REAL prevError = 0.0;
	uint32_t l = 0u;
	while (l < 100u)
	{
		d_error.front() = static_cast<REAL>(0.0);

		ComputeJacobiViscosity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_error.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_error = d_error;
		//h_error.front() /= static_cast<REAL>(sph.GetSize() + DBL_EPSILON);
		if (h_error.front() < 1.0e-3 && l >= 0u) break;
		{
			std::stringstream ss;
			ss << "Implicit Viscosity Error " << l << " : " << h_error.front() << std::endl;
			OutputDebugStringA(ss.str().c_str());
		}
		omega = omega * ((prevError > 1.0e-5) ? std::min(prevError / h_error.front(), 1.0) : 1.0);
		//omega = std::min((prevError > 1.0e-5) ? std::min(prevError / h_error.front(), 1.0) : 1.0, omega);
		prevError = h_error.front();

		ApplyJacobiViscosity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(sph, omega);
		CUDA_CHECK(cudaPeekAtLastError());
		l++;
	}

	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVel()), thrust::device_pointer_cast(sph.GetPredictVel() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
}

void mps::kernel::sph::ApplyImplicitGDViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL3> d_R(sph.GetSize());
	thrust::host_vector<REAL3> h_R(sph.GetSize());
	thrust::device_vector<REAL> d_gama(2);
	thrust::host_vector<REAL> h_gama;

	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPredictVel()));

	uint32_t l = 0u;
	while (l < 100u)
	{
		ComputeGDViscosityR_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_R.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		thrust::fill(d_gama.begin(), d_gama.end(), static_cast<REAL>(0.0));
		UpdateGDViscosityGama_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * 2 * sizeof(REAL) >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_gama.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_gama = d_gama;
		if (h_gama[0] < static_cast<REAL>(1.0e-2)) break;

	#if DEBUG_PRINT
		std::stringstream ss;
		ss << "Implicit GD Viscosity Error " << l << " : " << h_gama[0] << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif

		const auto gama = h_gama[0] / h_gama[1];
		UpdateGDViscosity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(sph, thrust::raw_pointer_cast(d_R.data()), gama);
		CUDA_CHECK(cudaPeekAtLastError());
		l++;
	}

	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVel()), thrust::device_pointer_cast(sph.GetPredictVel() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
}

void mps::kernel::sph::ApplyImplicitCGViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL3> d_R(sph.GetSize());
	thrust::device_vector<REAL3> d_V(sph.GetSize());
	thrust::device_vector<REAL3> d_Av(sph.GetSize());
	thrust::device_vector<REAL> d_param(2);
	thrust::host_vector<REAL> h_param;

	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPredictVel()));
	ComputeGDViscosityR_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(physParam, sph, material, hash, thrust::raw_pointer_cast(d_R.data()));
	CUDA_CHECK(cudaPeekAtLastError());
	thrust::copy(d_R.begin(), d_R.end(), d_V.begin());

	uint32_t l = 0u;
	while (l < 100u)
	{
		ComputeCGViscosityAv_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		thrust::fill(d_param.begin(), d_param.end(), static_cast<REAL>(0.0));
		UpdateCGViscosityAlphaParam_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * 2 * sizeof(REAL) >> >
			(sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), thrust::raw_pointer_cast(d_param.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_param = d_param;
		if (h_param[0] < static_cast<REAL>(1.0e-2)) break;

	#if DEBUG_PRINT
		std::stringstream ss;
		ss << "Implicit GD Viscosity Error " << l << " : " << h_param[0] << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif

		const auto alpha = h_param[0] / h_param[1];
		UpdateCGViscosityXR_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), alpha);
		CUDA_CHECK(cudaPeekAtLastError());

		d_param[0] = static_cast<REAL>(0.0);
		UpdateCGViscosityBetaParam_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> >
			(sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_param.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_param[1] = d_param[0];
		const auto beta = h_param[1] / h_param[0];
		UpdateCGViscosityV_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), beta);
		CUDA_CHECK(cudaPeekAtLastError());
		l++;
	}

	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVel()), thrust::device_pointer_cast(sph.GetPredictVel() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
}

void mps::kernel::sph::DensityColorTest(const mps::SPHParam& sph, const mps::SPHMaterialParam& material)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	DensityColorTest_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material);
	CUDA_CHECK(cudaPeekAtLastError());
}