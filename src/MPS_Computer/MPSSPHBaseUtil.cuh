#pragma once

#include "MPSSPHBaseUtil.h"
#include "MPSSPHKernel.cuh"

#include "../MPS_Object/MPSUniformDef.h"
#include "../MPS_Object/MPSSPHMaterial.h"
#include "../MPS_Object/MPSMeshMaterial.h"

__global__ void ComputeBoundaryParticleVolume_kernel(
	const REAL3* MCUDA_RESTRICT pBoundaryParticlePosition,
	const REAL* MCUDA_RESTRICT pBoundaryParticleRadius,
	REAL* MCUDA_RESTRICT pBoundaryParticleVolume,
	size_t boundaryParticleSize,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= boundaryParticleSize) return;

	const auto hi = pBoundaryParticleRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pBoundaryParticlePosition[id];

	REAL volume = mps::device::SPH::WKernel(0.0, invHi);

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pBoundaryParticleRadius[id] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pBoundaryParticlePosition[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		volume += mps::device::SPH::WKernel(dist, invHi);
	});

	mcuda::util::AtomicAdd(pBoundaryParticleVolume + id, volume);
}

__global__ void ComputeBoundaryParticleVolume_kernel(
	const REAL3* MCUDA_RESTRICT pBoundaryParticlePosition,
	const REAL* MCUDA_RESTRICT pBoundaryParticleRadius,
	REAL* MCUDA_RESTRICT pBoundaryParticleVolume,
	size_t boundaryParticleSize,
	const REAL3* MCUDA_RESTRICT pRefBoundaryParticlePosition,
	const REAL* MCUDA_RESTRICT pRefBoundaryParticleRadius,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= boundaryParticleSize) return;

	const auto hi = pBoundaryParticleRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pBoundaryParticlePosition[id];

	REAL volume = 0.0;

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pRefBoundaryParticleRadius[id] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pRefBoundaryParticlePosition[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		volume += mps::device::SPH::WKernel(dist, invHi);
	});

	mcuda::util::AtomicAdd(pBoundaryParticleVolume + id, volume);
}

__global__ void ComputeBoundaryParticleVolumeFinal_kernel(
	REAL* MCUDA_RESTRICT pBoundaryParticleVolume,
	size_t boundaryParticleSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= boundaryParticleSize) return;

	pBoundaryParticleVolume[id] = 1.0 / pBoundaryParticleVolume[id];
}

__global__ void ComputeDensity_kernel(
	const mps::SPHMaterialParam sphMaterial,
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	REAL* MCUDA_RESTRICT pSPHDensity,
	size_t sphSize,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];

	REAL density = sphMaterial.volume * mps::device::SPH::WKernel(0.0, invHi);

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pSPHPosition[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		density += sphMaterial.volume * mps::device::SPH::WKernel(dist, invHi);
	});

	mcuda::util::AtomicAdd(pSPHDensity + id, density);
}

__global__ void ComputeDensity_kernel(
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	REAL* MCUDA_RESTRICT pSPHDensity,
	size_t sphSize,
	const mps::SPHMaterialParam refSPHMaterial,
	const REAL3* MCUDA_RESTRICT pRefSPHPosition,
	const REAL* MCUDA_RESTRICT pRefSPHRadius,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];

	REAL density = 0.0;

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pRefSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pRefSPHPosition[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		density += refSPHMaterial.volume * mps::device::SPH::WKernel(dist, invHi);
	});

	mcuda::util::AtomicAdd(pSPHDensity + id, density);
}

__global__ void ComputeDensity_kernel(
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	REAL* MCUDA_RESTRICT pSPHDensity,
	size_t sphSize,
	const REAL3* MCUDA_RESTRICT pBoundaryParticlePosition,
	const REAL* MCUDA_RESTRICT pBoundaryParticleRadius,
	const REAL* MCUDA_RESTRICT pBoundaryParticleVolume,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];

	REAL density = 0.0;

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pBoundaryParticleRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pBoundaryParticlePosition[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		density += pBoundaryParticleVolume[jd] * mps::device::SPH::WKernel(dist, invHi);
	});

	mcuda::util::AtomicAdd(pSPHDensity + id, density);
}

__global__ void ComputeDensityFinal_kernel(
	const mps::SPHMaterialParam sphMaterial,
	REAL* MCUDA_RESTRICT pSPHDensity,
	size_t sphSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	pSPHDensity[id] = pSPHDensity[id] * sphMaterial.density;
}

__global__ void DensityColorTest_kernel(
	const mps::SPHMaterialParam sphMaterial,
	const REAL* MCUDA_RESTRICT pSPHDensity,
	glm::fvec4* MCUDA_RESTRICT pSPHColor,
	size_t sphSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto density = pSPHDensity[id];
	const float ratio = static_cast<float>(density / sphMaterial.density);

	const auto blue = mcuda::util::clamp(1.0f - ratio, 0.0f, 1.0f);
	const auto green = mcuda::util::clamp(ratio < 1.0f ? ratio : 1.0f - (ratio - 1.0f), 0.0f, 1.0f);
	const auto red = mcuda::util::clamp(ratio - 1.0f, 0.0f, 1.0f);
	pSPHColor[id] = {red, green, blue, 1.0f};
}