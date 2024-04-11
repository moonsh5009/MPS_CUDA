#include "stdafx.h"
#include "SPHUtil.cuh"
#include "SPHSurfaceTension.cuh"
#include <thrust/extrema.h>
#include "thrust/host_vector.h"
#include "AdvectUtil.h"

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

void mps::kernel::sph::DensityColorTest(const mps::SPHParam& sph, const mps::SPHMaterialParam& material)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	DensityColorTest_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material);
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

void mps::kernel::sph::ComputePressureForce(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL> d_error(1);
	thrust::host_vector<REAL> h_error(1);

	uint32_t l = 0u;
	while (l < 100u)
	{
		d_error.front() = static_cast<REAL>(0.0);

		ComputeCDPressure_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_error.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_error = d_error;
		h_error.front() / static_cast<REAL>(sph.GetSize() + DBL_EPSILON);
		if (h_error.front() < 1.0e-4 && l >= 2u) break;

		mps::kernel::ResetForce(sph);
		ApplyDFSPH(sph, material, hash);
		mps::kernel::UpdateVelocity(physParam, sph);
		l++;
	}
	printf("Pressure Force iteration %d\n", l);
}

void mps::kernel::sph::ComputeDivergenceFree(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL> d_error(1);
	thrust::host_vector<REAL> h_error(1);

	uint32_t l = 0u;
	while (l < 100u)
	{
		d_error.front() = static_cast<REAL>(0.0);

		ComputeDFPressure_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_error.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_error = d_error;
		h_error.front() / static_cast<REAL>(sph.GetSize() + DBL_EPSILON);
		if (h_error.front() < 1.0e-3 && l >= 1u) break;

		mps::kernel::ResetForce(sph);
		ApplyDFSPH(sph, material, hash);
		mps::kernel::UpdateVelocity(physParam, sph);
		l++;
	}
	printf("Divergence Free iteration %d\n", l);
}

void mps::kernel::sph::ApplyDFSPH(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ApplyDFSPH_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ApplyViscosity(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ApplyViscosity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ComputeColorField(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeColorField_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ApplySurfaceTension(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeColorField_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());

	/*ComputeLargeSmallDensity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(physParam, sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());*/

	ComputeLargeSmallDPressure_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(physParam, sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());

	auto dMaxColorField = thrust::max_element(thrust::device_ptr<REAL>(sph.GetColorFieldArray()), thrust::device_ptr<REAL>(sph.GetColorFieldArray() + sph.GetSize()));
	auto dMaxSmallPressure = thrust::max_element(thrust::device_ptr<REAL>(sph.GetSmallPressureArray()), thrust::device_ptr<REAL>(sph.GetSmallPressureArray() + sph.GetSize()));
	ComputeSurfaceTensor_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash, thrust::raw_pointer_cast(dMaxColorField), thrust::raw_pointer_cast(dMaxSmallPressure));
	CUDA_CHECK(cudaPeekAtLastError());

	//ComputeDensity(sph, material, hash);
	ApplySurfaceTension_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}