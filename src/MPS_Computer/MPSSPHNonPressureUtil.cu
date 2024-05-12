#include "stdafx.h"
#include "MPSSPHNonPressureUtil.cuh"
#include <thrust/extrema.h>

#include "../MPS_Object/MPSSPHParam.h"
#include "../MPS_Object/MPSBoundaryParticleParam.h"
#include "../MPS_Object/MPSSpatialHash.h"

#include "MPSSolver.cuh"

void mps::kernel::SPH::ComputeSurfaceTensionFactor(
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& nei)
{
	if (sph.size == 0) return;

	ComputeSurfaceTensionFactor_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sphMaterial,
		sph.pPosition,
		sph.pVelocity,
		sph.pRadius,
		sph.pMass,
		sph.pFactorST,
		sph.size,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeSurfaceTensionCGb(
	const mps::PhysicsParam& physParam,
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& nei)
{
	if (sph.size == 0) return;

	ComputeSurfaceTensionCGb_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		physParam,
		sphMaterial,
		sph.pPosition,
		sph.pVelocity,
		sph.pRadius,
		sph.pMass,
		sph.pDensity,
		sph.pFactorST,
		sph.size,
		nei.pID,
		nei.pIdx,
		sph.pTempVec3);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeSurfaceTensionCGAp(
	const mps::PhysicsParam& physParam,
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	REAL3* pCGp,
	REAL3* pCGAp)
{
	if (sph.size == 0) return;

	ComputeSurfaceTensionCGAp_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		physParam,
		sphMaterial,
		sph.pPosition,
		sph.pVelocity,
		sph.pRadius,
		sph.pMass,
		sph.pDensity,
		sph.pFactorST,
		sph.size,
		nei.pID,
		nei.pIdx,
		pCGp,
		pCGAp);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeViscosityCGAp(
	const mps::PhysicsParam& physParam,
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	REAL3* pCGp,
	REAL3* pCGAp)
{
	if (sph.size == 0) return;

	ComputeCGViscosityAp_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		physParam,
		sphMaterial,
		sph.pPosition,
		sph.pRadius,
		sph.pMass,
		sph.pDensity,
		sph.size,
		nei.pID,
		nei.pIdx,
		pCGp,
		pCGAp);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeFinalCGAp(
	const mps::PhysicsParam& physParam,
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	REAL3* pCGp,
	REAL3* pCGAp,
	REAL factor)
{
	if (sph.size == 0) return;

	ComputeFinalCGAp_kernel << < mcuda::util::DivUp(sph.size, nFullBlockSize), nFullBlockSize >> > (
		physParam,
		sph.size,
		pCGp,
		pCGAp,
		factor);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ApplyImplicitViscosityNSurfaceTension(
	const mps::PhysicsParam& physParam,
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& nei)
{
	if (sph.size == 0) return;

#if SPH_TIMER_PRINT
	cudaDeviceSynchronize();
	MTimer::Start("ApplyImplicitViscosityNSurfaceTension");
#endif

	thrust::device_vector<REAL3> d_r(sph.size);
	thrust::device_vector<REAL3> d_p(sph.size);
	thrust::device_vector<REAL3> d_Ap(sph.size);

	thrust::copy(
		thrust::device_pointer_cast(sph.pVelocity),
		thrust::device_pointer_cast(sph.pVelocity + sph.size),
		thrust::device_pointer_cast(sph.pPredictVel));

	ComputeSurfaceTensionFactor(sphMaterial, sph, nei);
	ComputeSurfaceTensionCGb(physParam, sphMaterial, sph, nei);

	const auto l = mps::kernel::CGSolver<REAL>(
		reinterpret_cast<REAL*>(thrust::raw_pointer_cast(d_r.data())),
		reinterpret_cast<REAL*>(thrust::raw_pointer_cast(d_p.data())),
		reinterpret_cast<REAL*>(thrust::raw_pointer_cast(d_Ap.data())),
		reinterpret_cast<REAL*>(sph.pPredictVel),
		reinterpret_cast<REAL*>(sph.pTempVec3),
		static_cast<REAL>(1.0e-2),
		sph.size * 3,
		100u,
		[&](REAL* p, REAL* Ap, uint32_t l)
	{
		auto pCGp = reinterpret_cast<REAL3*>(p);
		auto pCGAp = reinterpret_cast<REAL3*>(Ap);
		REAL factor = [&]
		{
			if (l == 0u) return static_cast<REAL>(0.0);
			/*if (l < 18u)		factor = static_cast<T>(0.01);
			else if (l == 18u)	factor = static_cast<T>(0.1);
			else				factor += l * static_cast<T>(0.016);*/
			return static_cast<REAL>(0.01);
		}();

		thrust::fill(thrust::device_pointer_cast(pCGAp), thrust::device_pointer_cast(pCGAp + sph.size), REAL3{ 0.0 });
		ComputeSurfaceTensionCGAp(physParam, sphMaterial, sph, nei, pCGp, pCGAp);
		ComputeViscosityCGAp(physParam, sphMaterial, sph, nei, pCGp, pCGAp);
		ComputeFinalCGAp(physParam, sphMaterial, sph, nei, pCGp, pCGAp, factor);
	});

	thrust::copy(
		thrust::device_pointer_cast(sph.pPredictVel),
		thrust::device_pointer_cast(sph.pPredictVel + sph.size),
		thrust::device_pointer_cast(sph.pVelocity));

#if SPH_TIMER_PRINT
	cudaDeviceSynchronize();
	std::stringstream ss;
	ss << "Loop " << " : " << l;
	MTimer::EndWithMessage("ApplyImplicitViscosityNSurfaceTension", ss.str());
#endif
}
