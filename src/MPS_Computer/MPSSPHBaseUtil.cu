#include "stdafx.h"
#include "MPSSPHBaseUtil.cuh"
#include <thrust/extrema.h>

#include "../MPS_Object/MPSSPHParam.h"
#include "../MPS_Object/MPSBoundaryParticleParam.h"
#include "../MPS_Object/MPSSpatialHash.h"

void mps::kernel::SPH::ComputeBoundaryParticleVolumeSub(
	const mps::BoundaryParticleParam& boundaryParticle,
	const mps::NeiParam& nei)
{
	if (boundaryParticle.size == 0) return;

	ComputeBoundaryParticleVolume_kernel << < mcuda::util::DivUp(boundaryParticle.size, nBlockSize), nBlockSize >> > (
		boundaryParticle.pPosition,
		boundaryParticle.pRadius,
		boundaryParticle.pVolume,
		boundaryParticle.size,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeBoundaryParticleVolumeSub(
	const mps::BoundaryParticleParam& boundaryParticle,
	const mps::NeiParam& nei,
	const mps::BoundaryParticleParam& refBoundaryParticle)
{
	if (boundaryParticle.size == 0) return;

	ComputeBoundaryParticleVolume_kernel << < mcuda::util::DivUp(boundaryParticle.size, nBlockSize), nBlockSize >> > (
		boundaryParticle.pPosition,
		boundaryParticle.pRadius,
		boundaryParticle.pVolume,
		boundaryParticle.size,
		refBoundaryParticle.pPosition,
		refBoundaryParticle.pRadius,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeBoundaryParticleVolumeFinal(
	const mps::BoundaryParticleParam& boundaryParticle)
{
	if (boundaryParticle.size == 0) return;

	ComputeBoundaryParticleVolumeFinal_kernel << < mcuda::util::DivUp(boundaryParticle.size, nFullBlockSize), nFullBlockSize >> > (
		boundaryParticle.pVolume,
		boundaryParticle.size);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDensitySub(
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph,
	const mps::NeiParam& nei)
{
	if (sph.size == 0) return;

	ComputeDensity_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sphMaterial,
		sph.pPosition,
		sph.pRadius,
		sph.pDensity,
		sph.size,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDensitySub(
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	const mps::SPHMaterialParam& refSPHMaterial,
	const mps::SPHParam& refSPH)
{
	if (sph.size == 0) return;

	ComputeDensity_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sph.pPosition,
		sph.pRadius,
		sph.pDensity,
		sph.size,
		refSPHMaterial,
		refSPH.pPosition,
		refSPH.pRadius,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDensitySub(
	const mps::SPHParam& sph,
	const mps::NeiParam& nei,
	const mps::BoundaryParticleParam& boundaryParticle)
{
	if (sph.size == 0) return;

	ComputeDensity_kernel << < mcuda::util::DivUp(sph.size, nBlockSize), nBlockSize >> > (
		sph.pPosition,
		sph.pRadius,
		sph.pDensity,
		sph.size,
		boundaryParticle.pPosition,
		boundaryParticle.pRadius,
		boundaryParticle.pVolume,
		nei.pID,
		nei.pIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::ComputeDensityFinal(
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph)
{
	if (sph.size == 0) return;

	ComputeDensityFinal_kernel << < mcuda::util::DivUp(sph.size, nFullBlockSize), nFullBlockSize >> > (
		sphMaterial,
		sph.pDensity,
		sph.size);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::SPH::DensityColorTest(
	const mps::SPHMaterialParam& sphMaterial,
	const mps::SPHParam& sph)
{
	if (sph.size == 0) return;

	DensityColorTest_kernel << < mcuda::util::DivUp(sph.size, nFullBlockSize), nFullBlockSize >> > (
		sphMaterial,
		sph.pDensity,
		sph.pColor,
		sph.size);
	CUDA_CHECK(cudaPeekAtLastError());
}