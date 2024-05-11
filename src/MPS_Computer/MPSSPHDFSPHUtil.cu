#include "stdafx.h"
#include "MPSSPHDFSPHUtil.cuh"
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

#include "../MPS_Object/MPSSPHParam.h"
#include "../MPS_Object/MPSBoundaryParticleParam.h"
#include "../MPS_Object/MPSSpatialHash.h"

#include "MPSAdvectUtil.h"

void mps::kernel::SPH::ComputeDFSPHFactorSub(
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& nei)
{
	if (sph.size == 0) return;
	
	ComputeDFSPHFactor_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sphMaterial,
		sph.pPosition,
		sph.pRadius,
		sph.pTempVec3,
		sph.pFactorDFSPH,
		sph.size,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDFSPHFactorSub(
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	const mps::SPHMaterialParam& refSPHMaterial,
	const mps::SPHParam& refSPH)
{
	if (sph.size == 0) return;

	ComputeDFSPHFactor_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sph.pPosition,
		sph.pRadius,
		sph.pTempVec3,
		sph.pFactorDFSPH,
		sph.size,
		refSPHMaterial,
		refSPH.pPosition,
		refSPH.pRadius,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDFSPHFactorSub(
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	const mps::BoundaryParticleParam& boundaryParticle)
{
	if (sph.size == 0) return;

	ComputeDFSPHFactor_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sph.pPosition,
		sph.pRadius,
		sph.pTempVec3,
		sph.size,
		boundaryParticle.pPosition,
		boundaryParticle.pRadius,
		boundaryParticle.pVolume,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDFSPHFactorFinal(
	const mps::SPHParam& sph)
{
	if (sph.size == 0) return;

	ComputeDFSPHFactorFinal_kernel << < mcuda::util::DivUp(sph.size, nFullBlockSize), nFullBlockSize >> > (
		sph.pTempVec3,
		sph.pFactorDFSPH,
		sph.size);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDensityDeltaSub(
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& nei)
{
	if (sph.size == 0) return;

	ComputeDensityDelta_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sphMaterial,
		sph.pPosition,
		sph.pVelocity,
		sph.pRadius,
		sph.pTempReal,
		sph.size,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDensityDeltaSub(
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	const mps::SPHMaterialParam& refSPHMaterial,
	const mps::SPHParam& refSPH)
{
	if (sph.size == 0) return;

	ComputeDensityDelta_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sph.pPosition,
		sph.pVelocity,
		sph.pRadius,
		sph.pTempReal,
		sph.size,
		refSPHMaterial,
		refSPH.pPosition,
		refSPH.pVelocity,
		refSPH.pRadius,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDensityDeltaSub(
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	const mps::BoundaryParticleParam& boundaryParticle)
{
	if (sph.size == 0) return;

	ComputeDensityDelta_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sph.pPosition,
		sph.pVelocity,
		sph.pRadius,
		sph.pTempReal,
		sph.size,
		boundaryParticle.pPosition,
		boundaryParticle.pVelocity,
		boundaryParticle.pRadius,
		boundaryParticle.pVolume,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDFSPHConstantDensitySub(
	const mps::PhysicsParam& physParam,
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	REAL* sumError)
{
	if (sph.size == 0) return;

	ComputeCDStiffness_kernel << < mcuda::util::DivUp(sph.size, nFullBlockSize), nFullBlockSize, nFullBlockSize * sizeof(REAL) >> > (
		physParam,
		sphMaterial,
		sph.pDensity,
		sph.pFactorDFSPH,
		sph.pTempReal,
		sph.pPressure,
		sph.size,
		sumError);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDFSPHDivergenceFreeSub(
	const mps::PhysicsParam& physParam,
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& neiSPH2SPH,
	const mps::NeiParam& neiSPH2BoundaryParticle,
	REAL* sumError)
{
	if (sph.size == 0) return;

	ComputeDFStiffness_kernel << < mcuda::util::DivUp(sph.size, nFullBlockSize), nFullBlockSize, nFullBlockSize * sizeof(REAL) >> > (
		physParam,
		sphMaterial,
		sph.pDensity,
		sph.pFactorDFSPH,
		sph.pTempReal,
		sph.pPressure,
		sph.size,
		neiSPH2SPH.pIdx,
		neiSPH2BoundaryParticle.pIdx,
		sumError);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ApplyDFSPHSub(
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& nei)
{
	if (sph.size == 0) return;

	ApplyDFSPH_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sphMaterial,
		sph.pPosition,
		sph.pRadius,
		sph.pPressure,
		sph.pForce,
		sph.size,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ApplyDFSPHSub(
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	const mps::SPHMaterialParam& pRefSPHMaterial,
	const mps::SPHParam& refSPH)
{
	if (sph.size == 0) return;

	ApplyDFSPH_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sph.pPosition,
		sph.pRadius,
		sph.pPressure,
		sph.pForce,
		sph.size,
		pRefSPHMaterial,
		refSPH.pPosition,
		refSPH.pRadius,
		refSPH.pPressure,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ApplyDFSPHSub(
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	const mps::BoundaryParticleParam& boundaryParticle)
{
	if (sph.size == 0) return;

	ApplyDFSPH_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sph.pPosition,
		sph.pRadius,
		sph.pPressure,
		sph.pForce,
		sph.size,
		boundaryParticle.pPosition,
		boundaryParticle.pRadius,
		boundaryParticle.pVolume,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ApplyDFSPHFinal(
	const mps::SPHParam& sph)
{
	if (sph.size == 0) return;

	ApplyDFSPHFinal_kernel << < mcuda::util::DivUp(sph.size, nFullBlockSize), nFullBlockSize >> > (
		sph.pMass,
		sph.pForce,
		sph.size);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDFSPHConstantDensity(
	const mps::PhysicsParam& physParam,
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::MeshMaterialParam& boundaryParticleMaterial,
	const mps::BoundaryParticleParam& boundaryParticle,
	const mps::NeiParam& neiSPH2SPH,
	const mps::NeiParam& neiSPH2BoundaryParticle)
{
	if (sph.size == 0) return;

#if SPH_TIMER_PRINT
	cudaDeviceSynchronize();
	MTimer::Start("mps::kernel::SPH::ComputeDFSPHConstantDensity");
#endif

	thrust::device_vector<REAL> d_error(1);
	thrust::host_vector<REAL> h_error(1);

	uint32_t l = 1u;
	while (l <= 100u)
	{
		d_error.front() = static_cast<REAL>(0.0);

		thrust::fill(
			thrust::device_pointer_cast(sph.pTempReal),
			thrust::device_pointer_cast(sph.pTempReal + sph.size),
			static_cast<REAL>(0.0));
		ComputeDensityDeltaSub(sphMaterial, sph, neiSPH2SPH);
		ComputeDensityDeltaSub(sph, neiSPH2BoundaryParticle, boundaryParticle);
		ComputeDFSPHConstantDensitySub(physParam, sphMaterial, sph, thrust::raw_pointer_cast(d_error.data()));

		h_error = d_error;
		h_error.front() /= static_cast<REAL>(sph.size);
		if (h_error.front() < 1.0e-4 && l > 2u) break;

	#if SPH_DEBUG_PRINT
		std::stringstream ss;
		ss << "Pressure Error " << l << " : " << h_error.front() << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif

		thrust::fill(thrust::device_pointer_cast(sph.pForce), thrust::device_pointer_cast(sph.pForce + sph.size), REAL3{ 0.0 });
		ApplyDFSPHSub(sphMaterial, sph, neiSPH2SPH);
		ApplyDFSPHSub(sph, neiSPH2BoundaryParticle, boundaryParticle);
		ApplyDFSPHFinal(sph);
		mps::kernel::Advect::UpdateVelocity(physParam, sph);
		l++;
	}

#if SPH_TIMER_PRINT
	cudaDeviceSynchronize();
	MTimer::End("mps::kernel::SPH::ComputeDFSPHConstantDensity");
#endif
}
void mps::kernel::SPH::ComputeDFSPHDivergenceFree(
	const mps::PhysicsParam& physParam,
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::MeshMaterialParam& boundaryParticleMaterial,
	const mps::BoundaryParticleParam& boundaryParticle,
	const mps::NeiParam& neiSPH2SPH,
	const mps::NeiParam& neiSPH2BoundaryParticle)
{
	if (sph.size == 0) return;

#if SPH_TIMER_PRINT
	cudaDeviceSynchronize();
	MTimer::Start("mps::kernel::SPH::ComputeDFSPHDivergenceFree");
#endif

	thrust::device_vector<REAL> d_error(1);
	thrust::host_vector<REAL> h_error(1);

	uint32_t l = 1u;
	while (l <= 100u)
	{
		d_error.front() = static_cast<REAL>(0.0);

		thrust::fill(
			thrust::device_pointer_cast(sph.pTempReal),
			thrust::device_pointer_cast(sph.pTempReal + sph.size),
			static_cast<REAL>(0.0));
		ComputeDensityDeltaSub(sphMaterial, sph, neiSPH2SPH);
		ComputeDensityDeltaSub(sph, neiSPH2BoundaryParticle, boundaryParticle);
		ComputeDFSPHDivergenceFreeSub(physParam, sphMaterial, sph, neiSPH2SPH, neiSPH2BoundaryParticle, thrust::raw_pointer_cast(d_error.data()));

		h_error = d_error;
		h_error.front() /= static_cast<REAL>(sph.size);
		if (h_error.front() < 1.0e-3 && l > 1u) break;

	#if SPH_DEBUG_PRINT
		std::stringstream ss;
		ss << "DivergenceFree Error " << l << " : " << h_error.front() << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif

		thrust::fill(thrust::device_pointer_cast(sph.pForce), thrust::device_pointer_cast(sph.pForce + sph.size), REAL3{ 0.0 });
		ApplyDFSPHSub(sphMaterial, sph, neiSPH2SPH);
		ApplyDFSPHSub(sph, neiSPH2BoundaryParticle, boundaryParticle);
		ApplyDFSPHFinal(sph);
		mps::kernel::Advect::UpdateVelocity(physParam, sph);
		l++;
	}

#if SPH_TIMER_PRINT
	cudaDeviceSynchronize();
	MTimer::End("mps::kernel::SPH::ComputeDFSPHDivergenceFree");
#endif
}