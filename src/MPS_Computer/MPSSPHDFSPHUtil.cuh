#pragma once

#include "MPSSPHDFSPHUtil.h"
#include "MPSSPHKernel.cuh"

#include "../MPS_Object/MPSUniformDef.h"
#include "../MPS_Object/MPSSPHMaterial.h"
#include "../MPS_Object/MPSMeshMaterial.h"

__global__ void ComputeDFSPHFactor_kernel(
	mps::SPHMaterialParam sphMaterial,
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	REAL3* MCUDA_RESTRICT pSPHGradSum,
	REAL* MCUDA_RESTRICT pSPHFactorDFSPH,
	size_t sphSize,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];

	REAL3 grads{ 0.0 };
	REAL factorDFSPHi = 0.0;

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pSPHPosition[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		const auto grad = sphMaterial.volume * mps::device::SPH::GKernel(dist, invHi) * xij;
		grads += grad;
		factorDFSPHi += glm::dot(grad, grad);
	});

	mps::device::AtomicAdd(pSPHGradSum + id, grads);
	mcuda::util::AtomicAdd(pSPHFactorDFSPH + id, factorDFSPHi);
}

__global__ void ComputeDFSPHFactor_kernel(
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	REAL3* MCUDA_RESTRICT pSPHGradSum,
	REAL* MCUDA_RESTRICT pSPHFactorDFSPH,
	size_t sphSize,
	mps::SPHMaterialParam refSPHMaterial,
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

	REAL3 grads{ 0.0 };
	REAL factorDFSPHi = 0.0;

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pRefSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pRefSPHPosition[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		const auto grad = refSPHMaterial.volume * mps::device::SPH::GKernel(dist, invHi) * xij;
		grads += grad;
		factorDFSPHi += glm::dot(grad, grad);
	});

	mps::device::AtomicAdd(pSPHGradSum + id, grads);
	mcuda::util::AtomicAdd(pSPHFactorDFSPH + id, factorDFSPHi);
}

__global__ void ComputeDFSPHFactor_kernel(
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	REAL3* MCUDA_RESTRICT pSPHGradSum,
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

	REAL3 grads{ 0.0 };

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pBoundaryParticleRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pBoundaryParticlePosition[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		const auto grad = pBoundaryParticleVolume[jd] * mps::device::SPH::GKernel(dist, invHi) * xij;
		grads += grad;
	});

	mps::device::AtomicAdd(pSPHGradSum + id, grads);
}

__global__ void ComputeDFSPHFactorFinal_kernel(
	const REAL3* MCUDA_RESTRICT pSPHGradSum,
	REAL* MCUDA_RESTRICT pSPHFactorDFSPH,
	size_t sphSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	pSPHFactorDFSPH[id] += glm::dot(pSPHGradSum[id], pSPHGradSum[id]);
	pSPHFactorDFSPH[id] = pSPHFactorDFSPH[id] > mps::device::SPH::SPH_EPSILON ? 1.0 / pSPHFactorDFSPH[id] : 0.0;
}

__global__ void ComputeDensityDelta_kernel(
	mps::SPHMaterialParam sphMaterial,
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL3* MCUDA_RESTRICT pSPHVelocity,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	REAL* MCUDA_RESTRICT pSPHDelta,
	size_t sphSize,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];
	const auto vi = pSPHVelocity[id];

	REAL delta = 0.0;

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pSPHPosition[jd];
		const auto vj = pSPHVelocity[jd];

		const auto xij = xi - xj;
		const auto vij = vi - vj;
		const auto dist = glm::length(xij);
		delta += sphMaterial.volume * mps::device::SPH::GKernel(dist, invHi) * glm::dot(xij, vij);
	});

	mcuda::util::AtomicAdd(pSPHDelta + id, delta);
}

__global__ void ComputeDensityDelta_kernel(
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL3* MCUDA_RESTRICT pSPHVelocity,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	REAL* MCUDA_RESTRICT pSPHDelta,
	size_t sphSize,
	mps::SPHMaterialParam refSPHMaterial,
	const REAL3* MCUDA_RESTRICT pRefSPHPosition,
	const REAL3* MCUDA_RESTRICT pRefSPHVelocity,
	const REAL* MCUDA_RESTRICT pRefSPHRadius,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];
	const auto vi = pSPHVelocity[id];

	REAL delta = 0.0;

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pRefSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pRefSPHPosition[jd];
		const auto vj = pRefSPHVelocity[jd];

		const auto xij = xi - xj;
		const auto vij = vi - vj;
		const auto dist = glm::length(xij);
		delta += refSPHMaterial.volume * mps::device::SPH::GKernel(dist, invHi) * glm::dot(xij, vij);
	});

	mcuda::util::AtomicAdd(pSPHDelta + id, delta);
}

__global__ void ComputeDensityDelta_kernel(
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL3* MCUDA_RESTRICT pSPHVelocity,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	REAL* MCUDA_RESTRICT pSPHDelta,
	size_t sphSize,
	const REAL3* MCUDA_RESTRICT pBoundaryParticlePosition,
	const REAL3* MCUDA_RESTRICT pBoundaryParticleVelocity,
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
	const auto vi = pSPHVelocity[id];

	REAL delta = 0.0;

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pBoundaryParticleRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pBoundaryParticlePosition[jd];
		const auto vj = pBoundaryParticleVelocity[jd];

		const auto xij = xi - xj;
		const auto vij = vi - vj;
		const auto dist = glm::length(xij);
		delta += pBoundaryParticleVolume[jd] * mps::device::SPH::GKernel(dist, invHi) * glm::dot(xij, vij);
	});

	mcuda::util::AtomicAdd(pSPHDelta + id, delta);
}

__global__ void ComputeCDStiffness_kernel(
	mps::PhysicsParam physParam,
	mps::SPHMaterialParam sphMaterial,
	const REAL* MCUDA_RESTRICT pSPHDensity,
	const REAL* MCUDA_RESTRICT pSPHFactorDFSPH,
	const REAL* MCUDA_RESTRICT pSPHDelta,
	REAL* MCUDA_RESTRICT pSPHPressure,
	size_t sphSize,
	REAL* MCUDA_RESTRICT sumError)
{
	extern __shared__ REAL s_sumErrors[];
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	s_sumErrors[threadIdx.x] = 0u;
	if (id < sphSize)
	{
		auto stiffness = pSPHDensity[id] / sphMaterial.density + physParam.dt * pSPHDelta[id] - 1.0;
		if (stiffness > 0.0)
		{
			s_sumErrors[threadIdx.x] = stiffness;
			stiffness *= pSPHFactorDFSPH[id] / (physParam.dt * physParam.dt);
		}
		else stiffness = 0.0;

		pSPHPressure[id] = stiffness;
	}
#pragma unroll
	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
	{
		__syncthreads();
		if (threadIdx.x < s)
			s_sumErrors[threadIdx.x] += s_sumErrors[threadIdx.x + s];
	}
	__syncthreads();
	if (threadIdx.x < 32u)
	{
		mcuda::util::WarpSum(s_sumErrors, threadIdx.x);
		if (threadIdx.x == 0)
			mcuda::util::AtomicAdd(sumError, s_sumErrors[0]);
	}
}

__global__ void ComputeDFStiffness_kernel(
	mps::PhysicsParam physParam,
	mps::SPHMaterialParam sphMaterial,
	const REAL* MCUDA_RESTRICT pSPHDensity,
	const REAL* MCUDA_RESTRICT pSPHFactorDFSPH,
	const REAL* MCUDA_RESTRICT pSPHDelta,
	REAL* MCUDA_RESTRICT pSPHPressure,
	size_t sphSize,
	REAL* MCUDA_RESTRICT sumError)
{
	extern __shared__ REAL s_sumErrors[];
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	s_sumErrors[threadIdx.x] = 0u;
	if (id < sphSize)
	{
		auto stiffness = mcuda::util::min(pSPHDelta[id] * physParam.dt, pSPHDensity[id] / sphMaterial.density + physParam.dt * pSPHDelta[id] - 0.8);
		if (stiffness > 0.0)
		{
			s_sumErrors[threadIdx.x] = stiffness;
			stiffness *= pSPHFactorDFSPH[id] / (physParam.dt * physParam.dt);
		}
		else stiffness = 0.0;
		pSPHPressure[id] = stiffness;
	}
#pragma unroll
	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
	{
		__syncthreads();
		if (threadIdx.x < s)
			s_sumErrors[threadIdx.x] += s_sumErrors[threadIdx.x + s];
	}
	__syncthreads();
	if (threadIdx.x < 32u)
	{
		mcuda::util::WarpSum(s_sumErrors, threadIdx.x);
		if (threadIdx.x == 0)
			mcuda::util::AtomicAdd(sumError, s_sumErrors[0]);
	}
}

__global__ void ApplyDFSPH_kernel(
	mps::SPHMaterialParam sphMaterial,
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	const REAL* MCUDA_RESTRICT pSPHPressure,
	REAL3* MCUDA_RESTRICT pSPHForce,
	size_t sphSize,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];
	const auto pi = pSPHPressure[id];

	REAL3 force{ 0.0 };
	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pSPHPosition[jd];
		const auto pj = pSPHPressure[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		const auto forceij = 0.5 * sphMaterial.volume * (pi + pj) * mps::device::SPH::GKernel(dist, invHi) * xij;
		force -= forceij;
	});

	mps::device::AtomicAdd(pSPHForce + id, force);
}

__global__ void ApplyDFSPH_kernel(
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	const REAL* MCUDA_RESTRICT pSPHPressure,
	REAL3* MCUDA_RESTRICT pSPHForce,
	size_t sphSize,
	mps::SPHMaterialParam refSPHMaterial,
	const REAL3* MCUDA_RESTRICT pRefSPHPosition,
	const REAL* MCUDA_RESTRICT pRefSPHRadius,
	const REAL* MCUDA_RESTRICT pRefSPHPressure,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];
	const auto pi = pSPHPressure[id];

	REAL3 force{ 0.0 };

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pRefSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pRefSPHPosition[jd];
		const auto pj = pRefSPHPressure[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		const auto forceij = 0.5 * refSPHMaterial.volume * (pi + pj) * mps::device::SPH::GKernel(dist, invHi) * xij;
		force -= forceij;
	});

	mps::device::AtomicAdd(pSPHForce + id, force);
}

__global__ void ApplyDFSPH_kernel(
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	const REAL* MCUDA_RESTRICT pSPHPressure,
	REAL3* MCUDA_RESTRICT pSPHForce,
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
	const auto pi = pSPHPressure[id];

	REAL3 force{ 0.0 };

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pBoundaryParticleRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pBoundaryParticlePosition[jd];

		const auto xij = xi - xj;
		const auto dist = glm::length(xij);
		const auto forceij = 0.5 * pBoundaryParticleVolume[jd] * pi * mps::device::SPH::GKernel(dist, invHi) * xij;
		force -= forceij;
	});

	mps::device::AtomicAdd(pSPHForce + id, force);
}

__global__ void ApplyDFSPHFinal_kernel(
	const REAL* MCUDA_RESTRICT pSPHMass,
	REAL3* MCUDA_RESTRICT pSPHForce,
	size_t sphSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	pSPHForce[id] = pSPHMass[id] * pSPHForce[id];
}