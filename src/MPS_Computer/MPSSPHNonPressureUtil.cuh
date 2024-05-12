#pragma once

#include "MPSSPHNonPressureUtil.h"
#include "MPSSPHKernel.cuh"

#include "../MPS_Object/MPSUniformDef.h"
#include "../MPS_Object/MPSSPHMaterial.h"
#include "../MPS_Object/MPSMeshMaterial.h"

//__global__ void ApplyExplicitViscosity_kernel(
//	mps::SPHMaterialParam sphMaterial,
//	const REAL3* MCUDA_RESTRICT pSPHPosition,
//	const REAL* MCUDA_RESTRICT pSPHRadius,
//	REAL3* MCUDA_RESTRICT pSPHGradSum,
//	REAL* MCUDA_RESTRICT pSPHFactorDFSPH,
//	size_t sphSize,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sphSize) return;
//
//	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
//	const auto invHi = 1.0 / hi;
//	const auto onePercentHSq = 0.01 * hi * hi;
//
//	const auto xi = pSPHPosition[id];
//
//	REAL3 force{ 0.0 };
//
//	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto hj = pSPHRadius[jd] * mps::device::SPH::H_RATIO;
//		const auto invHj = 1.0 / hj;
//
//		const auto xj = pSPHPosition[jd];
//		const auto xij = xi - xj;
//
//		const auto dist = glm::length(xij);
//		const auto vj = sph.Velocity(jd);
//		const auto volumej = sph.Volume(jd);
//
//		const auto vij = vi - vj;
//		const auto forceij = volumej * glm::dot(vij, xij) / (dist * dist + onePercentHSq) * mps::device::SPH::GKernel(dist, invH) / (dist + FLT_EPSILON) * xij;
//		viscForce += forceij;
//	});
//
//	mps::device::AtomicAdd(pSPHGradSum + id, grads);
//	mcuda::util::AtomicAdd(pSPHFactorDFSPH + id, factorDFSPHi);
//	force *= 10.0 * material.viscosity * volumei * viscForce;
//	auto force = sph.Force(id);
//	force += 10.0 * material.viscosity * volumei * viscForce;
//	sph.Force(id) = force;
//}
//__global__ void ApplyExplicitSurfaceTension_kernel(
//	mps::SPHParam sph, const mps::SPHMaterialParam material,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto h = material.radius;
//	const auto invH = 1.0 / h;
//
//	const auto xi = sph.Position(id);
//	const auto volumei = sph.Volume(id);
//
//	REAL3 forceST{ 0.0 };
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto& xj = sph.Position(jd);
//		const auto xij = xi - xj;
//
//		const auto dist = glm::length(xij);
//		if (dist < h)
//		{
//			const auto volumej = sph.Volume(jd);
//			const auto forceij = volumej * mps::device::SPH::STWKernel(dist, invH) * xij;
//			forceST -= forceij;
//		}
//	});
//
//	auto force = sph.Force(id);
//	force += material.surfaceTension * volumei * forceST;
//	sph.Force(id) = force;
//}
//
//__global__ void ComputeJacobiViscosity_0_kernel(const mps::PhysicsParam physParam,
//	mps::SPHParam sph, const mps::SPHMaterialParam material,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx,
//	REAL3* MCUDA_RESTRICT vTemp)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto h = material.radius;
//	const auto invH = 1.0 / h;
//	const auto onePercentHSq = 0.01 * h * h;
//
//	const auto xi = sph.Position(id);
//
//	REAL3x3 Avi{ 0.0 };
//	REAL3 Avj{ 0.0 };
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = sph.Position(jd);
//		const auto xij = xi - xj;
//
//		const auto dist = glm::length(xij);
//		if (dist < h)
//		{
//			const auto predictVj = sph.PredictVel(jd);
//			const auto volumej = sph.Volume(jd);
//			const auto Aij = volumej * mps::device::SPH::GKernel(dist, invH) / ((dist * dist + onePercentHSq) * (dist + FLT_EPSILON)) * REAL3x3
//			{
//				xij.x* xij.x, xij.y* xij.x, xij.z* xij.x,
//					xij.x* xij.y, xij.y* xij.y, xij.z* xij.y,
//					xij.x* xij.z, xij.y* xij.z, xij.z* xij.z,
//			};
//			Avj -= predictVj * Aij;
//			Avi -= Aij;
//		}
//	});
//
//	const auto alpha = physParam.dt * 10.0 * material.viscosity / sph.Density(id);
//	Avj *= alpha;
//	Avi *= alpha;
//	Avi += REAL3x3{ 1.0 };
//	auto invAvi = glm::inverse(Avi);
//	if (isinf(invAvi[0][0]) || isnan(invAvi[0][0]))
//		invAvi = REAL3x3{ 0.0 };
//
//	const auto newVi = (sph.Velocity(id) + Avj) * invAvi;
//	vTemp[id] = newVi;
//}
//__global__ void ComputeJacobiViscosity_1_kernel(const mps::PhysicsParam physParam,
//	mps::SPHParam sph, const mps::SPHMaterialParam material,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx,
//	REAL3* MCUDA_RESTRICT vTemp0, REAL3* MCUDA_RESTRICT vTemp1, REAL3* MCUDA_RESTRICT vTemp2,
//	uint32_t l, REAL underRelax, REAL* MCUDA_RESTRICT omega)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto h = material.radius;
//	const auto invH = 1.0 / h;
//	const auto onePercentHSq = 0.01 * h * h;
//
//	const auto xi = sph.Position(id);
//
//	REAL3x3 Avi{ 0.0 };
//	REAL3 Avj{ 0.0 };
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = sph.Position(jd);
//		const auto xij = xi - xj;
//
//		const auto dist = glm::length(xij);
//		if (dist < h)
//		{
//			const auto predictVj = sph.PredictVel(jd);
//			const auto volumej = sph.Volume(jd);
//			const auto Aij = volumej * mps::device::SPH::GKernel(dist, invH) / ((dist * dist + onePercentHSq) * (dist + FLT_EPSILON)) * REAL3x3
//			{
//				xij.x* xij.x, xij.y* xij.x, xij.z* xij.x,
//					xij.x* xij.y, xij.y* xij.y, xij.z* xij.y,
//					xij.x* xij.z, xij.y* xij.z, xij.z* xij.z,
//			};
//			Avj -= predictVj * Aij;
//			Avi -= Aij;
//		}
//	});
//
//	const auto alpha = physParam.dt * 10.0 * material.viscosity / sph.Density(id);
//	Avj *= alpha;
//	Avi *= alpha;
//	Avi += REAL3x3{ 1.0 };
//	auto invAvi = glm::inverse(Avi);
//	if (isinf(invAvi[0][0]) || isnan(invAvi[0][0]))
//		invAvi = REAL3x3{ 0.0 };
//
//	const auto prevVi = sph.PreviousVel(id);
//	const auto currVi = sph.PredictVel(id);
//	const auto newVi = (sph.Velocity(id) + Avj) * invAvi;
//	vTemp0[id] = omega[0] * (underRelax * (newVi - currVi) + currVi - prevVi) + prevVi;
//	vTemp1[id] = omega[1] * (underRelax * (newVi - currVi) + currVi - prevVi) + prevVi;
//	vTemp2[id] = omega[2] * (underRelax * (newVi - currVi) + currVi - prevVi) + prevVi;
//}
//__global__ void ComputeJacobiError_0_kernel(const mps::PhysicsParam physParam,
//	mps::SPHParam sph, const mps::SPHMaterialParam material,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx,
//	REAL3* MCUDA_RESTRICT vTemp, REAL* MCUDA_RESTRICT sumError)
//{
//	extern __shared__ REAL s_sumErrors[];
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//
//	s_sumErrors[threadIdx.x] = 0.0;
//	if (id < sph.GetSize())
//	{
//		const auto h = material.radius;
//		const auto invH = 1.0 / h;
//		const auto onePercentHSq = 0.01 * h * h;
//
//		const auto xi = sph.Position(id);
//		const auto predictVi = vTemp[id];
//
//		REAL3 Avij{ 0.0 };
//
//		NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//		{
//			const auto xj = sph.Position(jd);
//			const auto xij = xi - xj;
//
//			const auto dist = glm::length(xij);
//			if (dist < h)
//			{
//				const auto predictVj = vTemp[jd];
//				const auto volumej = sph.Volume(jd);
//				const auto grad = volumej * mps::device::SPH::GKernel(dist, invH) / ((dist * dist + onePercentHSq) * (dist + FLT_EPSILON)) * xij;
//				Avij += glm::dot(xij, predictVi - predictVj) * grad;
//			}
//		});
//
//		const auto alpha = physParam.dt * 10.0 * material.viscosity / sph.Density(id);
//		Avij *= alpha;
//
//		const auto v0 = sph.Velocity(id);
//		s_sumErrors[threadIdx.x] = glm::length(predictVi - Avij - v0);
//	}
//#pragma unroll
//	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
//	{
//		__syncthreads();
//		if (threadIdx.x < s)
//			s_sumErrors[threadIdx.x] += s_sumErrors[threadIdx.x + s];
//	}
//	__syncthreads();
//	if (threadIdx.x < 32u)
//	{
//		mcuda::util::WarpSum(s_sumErrors, threadIdx.x);
//		if (threadIdx.x == 0)
//			mcuda::util::AtomicAdd(sumError, s_sumErrors[0]);
//	}
//}
//__global__ void ComputeJacobiError_1_kernel(const mps::PhysicsParam physParam,
//	mps::SPHParam sph, const mps::SPHMaterialParam material,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx,
//	REAL3* MCUDA_RESTRICT vTemp0, REAL3* MCUDA_RESTRICT vTemp1, REAL3* MCUDA_RESTRICT vTemp2, REAL* MCUDA_RESTRICT sumError)
//{
//	extern __shared__ REAL s_sumErrors[];
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//
//	s_sumErrors[threadIdx.x] = 0.0;
//	s_sumErrors[threadIdx.x + blockDim.x] = 0.0;
//	s_sumErrors[threadIdx.x + (blockDim.x << 1u)] = 0.0;
//	if (id < sph.GetSize())
//	{
//		const auto h = material.radius;
//		const auto invH = 1.0 / h;
//		const auto onePercentHSq = 0.01 * h * h;
//
//		const auto xi = sph.Position(id);
//		const auto predictVi0 = vTemp0[id];
//		const auto predictVi1 = vTemp1[id];
//		const auto predictVi2 = vTemp2[id];
//
//		REAL3 Avij0{ 0.0 };
//		REAL3 Avij1{ 0.0 };
//		REAL3 Avij2{ 0.0 };
//
//		NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//		{
//			const auto xj = sph.Position(jd);
//			const auto xij = xi - xj;
//
//			const auto dist = glm::length(xij);
//			if (dist < h)
//			{
//				const auto predictVj0 = vTemp0[jd];
//				const auto predictVj1 = vTemp1[jd];
//				const auto predictVj2 = vTemp2[jd];
//				const auto volumej = sph.Volume(jd);
//				const auto grad = volumej * mps::device::SPH::GKernel(dist, invH) / ((dist * dist + onePercentHSq) * (dist + FLT_EPSILON)) * xij;
//				Avij0 += glm::dot(xij, predictVi0 - predictVj0) * grad;
//				Avij1 += glm::dot(xij, predictVi1 - predictVj1) * grad;
//				Avij2 += glm::dot(xij, predictVi2 - predictVj2) * grad;
//			}
//		});
//
//		const auto alpha = physParam.dt * 10.0 * material.viscosity / sph.Density(id);
//		Avij0 *= alpha;
//		Avij1 *= alpha;
//		Avij2 *= alpha;
//
//		const auto v0 = sph.Velocity(id);
//		s_sumErrors[threadIdx.x] = glm::length(predictVi0 - Avij0 - v0);
//		s_sumErrors[threadIdx.x + blockDim.x] = glm::length(predictVi1 - Avij1 - v0);
//		s_sumErrors[threadIdx.x + (blockDim.x << 1u)] = glm::length(predictVi2 - Avij2 - v0);
//	}
//#pragma unroll
//	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
//	{
//		__syncthreads();
//		if (threadIdx.x < s)
//		{
//			s_sumErrors[threadIdx.x] += s_sumErrors[threadIdx.x + s];
//			s_sumErrors[threadIdx.x + blockDim.x] += s_sumErrors[threadIdx.x + blockDim.x + s];
//			s_sumErrors[threadIdx.x + (blockDim.x << 1u)] += s_sumErrors[threadIdx.x + (blockDim.x << 1u) + s];
//		}
//	}
//
//	__syncthreads();
//	if (threadIdx.x < 32u)
//	{
//		mcuda::util::WarpSum(s_sumErrors, threadIdx.x);
//		mcuda::util::WarpSum(s_sumErrors + blockDim.x, threadIdx.x);
//		mcuda::util::WarpSum(s_sumErrors + (blockDim.x << 1u), threadIdx.x);
//		if (threadIdx.x == 0)
//		{
//			mcuda::util::AtomicAdd(sumError, s_sumErrors[0]);
//			mcuda::util::AtomicAdd(sumError + 1, s_sumErrors[blockDim.x]);
//			mcuda::util::AtomicAdd(sumError + 2, s_sumErrors[(blockDim.x << 1u)]);
//		}
//	}
//}
//
//__global__ void ComputeJacobiViscosity_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash,
//	REAL3* vTemp, REAL omega, REAL underRelax, REAL* sumError)
//{
//	extern __shared__ REAL s_sumErrors[];
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//
//	s_sumErrors[threadIdx.x] = 0u;
//	if (id < sph.GetSize())
//	{
//		const auto h = material.radius;
//		const auto invH = 1.0 / h;
//		const auto onePercentHSq = 0.01 * h * h;
//
//		const auto xi = sph.Position(id);
//
//		REAL3x3 Avi{ 0.0 };
//		REAL3 Avj{ 0.0 };
//		hash.Research(xi, [&](uint32_t jd)
//		{
//			if (id == jd) return;
//
//			const auto xj = sph.Position(jd);
//			const auto xij = xi - xj;
//
//			const auto dist = glm::length(xij);
//			if (dist < h)
//			{
//				const auto predictVj = sph.PredictVel(jd);
//				const auto volumej = sph.Volume(jd);
//				const auto Aij = volumej * mps::kernel::SPH::GKernel(dist, invH) / ((dist * dist + onePercentHSq) * dist)
//					* REAL3x3
//				{
//					xij.x* xij.x, xij.y* xij.x, xij.z* xij.x,
//						xij.x* xij.y, xij.y* xij.y, xij.z* xij.y,
//						xij.x* xij.z, xij.y* xij.z, xij.z* xij.z,
//				};
//				Avj -= predictVj * Aij;
//				Avi -= Aij;
//			}
//		});
//		const auto alpha = physParam.dt * 10.0 * material.viscosity / sph.Density(id);
//		Avj *= alpha;
//		Avi *= alpha;
//		Avi += REAL3x3{ 1.0 };
//		auto invAvi = glm::inverse(Avi);
//		if (isinf(invAvi[0][0]) || isnan(invAvi[0][0]))
//			invAvi = REAL3x3{ 0.0 };
//
//		const auto GetErrorBetweenV = [](const REAL3& v1, const REAL3& v2)
//		{
//			return mcuda::util::max(mcuda::util::max(fabs(v1.x - v2.x), fabs(v1.y - v2.y)), fabs(v1.z - v2.z));
//		};
//		const auto prevVi = sph.PreviousVel(id);
//		const auto currVi = sph.PredictVel(id);
//		auto newVi = (sph.Velocity(id) + Avj) * invAvi;
//		newVi = omega * (underRelax * (newVi - currVi) + currVi - prevVi) + prevVi;
//		s_sumErrors[threadIdx.x] = GetErrorBetweenV(newVi, sph.PredictVel(id));
//		vTemp[id] = newVi;
//	}
//	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
//	{
//		__syncthreads();
//		if (threadIdx.x < s)
//		{
//			s_sumErrors[threadIdx.x] += s_sumErrors[threadIdx.x + s];
//		}
//	}
//	__syncthreads();
//	if (threadIdx.x < 32u)
//	{
//		mcuda::util::warpSum(s_sumErrors, threadIdx.x);
//		if (threadIdx.x == 0)
//		{
//			mcuda::util::AtomicAdd(sumError, s_sumErrors[0]);
//		}
//	}
//}
//
//__global__ void ApplyJacobiViscosity_kernel(mps::SPHParam sph, REAL3* vTemp)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto currVi = sph.PredictVel(id);
//	const auto newVi = vTemp[id];
//
//	sph.PreviousVel(id) = currVi;
//	sph.PredictVel(id) = newVi;
//}

__global__ void ComputeSurfaceTensionFactor_kernel(
	mps::SPHMaterialParam sphMaterial,
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL3* MCUDA_RESTRICT pSPHVelocity,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	const REAL* MCUDA_RESTRICT pSPHMass,
	REAL* MCUDA_RESTRICT pSPHFactorST,
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
		const auto mj = pSPHMass[jd];

		const auto xij = xi - xj;
		const auto vij = vi - vj;
		const auto dist = glm::length(xij);

		const auto grad = mps::device::SPH::GKernel(dist, invHi) / (dist + FLT_EPSILON) * xij;
		delta += mj * glm::dot(vij, grad);
	});

	pSPHFactorST[id] = delta;
}

__global__ void ComputeSurfaceTensionCGb_kernel(
	mps::PhysicsParam physParam,
	mps::SPHMaterialParam sphMaterial,
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL3* MCUDA_RESTRICT pSPHVelocity,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	const REAL* MCUDA_RESTRICT pSPHMass,
	const REAL* MCUDA_RESTRICT pSPHDensity,
	const REAL* MCUDA_RESTRICT pSPHFactorST,
	size_t sphSize,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx,
	REAL3* MCUDA_RESTRICT pCGb)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];
	const auto vi = pSPHVelocity[id];
	const auto mi = pSPHMass[id];
	const auto di = pSPHDensity[id];
	const auto vDeltai = pSPHFactorST[id];

	REAL3 bi{ 0.0 };

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pSPHPosition[jd];
		const auto vj = pSPHVelocity[jd];
		const auto mj = pSPHMass[jd];
		const auto dj = pSPHDensity[jd];
		const auto vDeltaj = pSPHFactorST[jd];

		const auto xij = xi - xj;
		const auto vij = vi - vj;
		const auto dist = glm::length(xij);

		const auto mij = 0.5 * (mi + mj);
		const auto invDij = 2.0 / (di + dj);

		const auto stW = mps::device::SPH::STWKernel(dist, invHi);
		const auto stG = mps::device::SPH::STGKernel(dist, invHi) / (dist + FLT_EPSILON) * xij;
		const auto gij = (stW + physParam.dt * (glm::dot(vij, stG) - (vDeltai + vDeltaj) * invDij * 0.5 * stW)) * invDij;

		bi -= mij * gij * xij;
	});

	pCGb[id] = physParam.dt * sphMaterial.surfaceTension / mi * bi + vi;
}

__global__ void ComputeSurfaceTensionCGAp_kernel(
	mps::PhysicsParam physParam,
	mps::SPHMaterialParam sphMaterial,
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL3* MCUDA_RESTRICT pSPHVelocity,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	const REAL* MCUDA_RESTRICT pSPHMass,
	const REAL* MCUDA_RESTRICT pSPHDensity,
	const REAL* MCUDA_RESTRICT pSPHFactorST,
	size_t sphSize,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx,
	const REAL3* MCUDA_RESTRICT pCGp,
	REAL3* MCUDA_RESTRICT pCGAp)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];
	const auto vi = pSPHVelocity[id];
	const auto mi = pSPHMass[id];
	const auto di = pSPHDensity[id];
	const auto vDeltai = pSPHFactorST[id];
	const auto pi = pCGp[id];

	REAL3 ApSTi{ 0.0 };
	REAL3 ApXSPHi{ 0.0 };

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;

		const auto xj = pSPHPosition[jd];
		const auto vj = pSPHVelocity[jd];
		const auto mj = pSPHMass[jd];
		const auto dj = pSPHDensity[jd];
		const auto vDeltaj = pSPHFactorST[jd];
		const auto pj = pCGp[jd];

		const auto xij = xi - xj;
		const auto vij = vi - vj;
		const auto dist = glm::length(xij);

		const auto mij = 0.5 * (mi + mj);
		const auto invDij = 2.0 / (di + dj);

		const auto stW = mps::device::SPH::STWKernel(dist, invHi);
		const auto stG = mps::device::SPH::STGKernel(dist, invHi) / (dist + FLT_EPSILON) * xij;
		const auto gij = (stW + physParam.dt * (glm::dot(vij, stG) - (vDeltai + vDeltaj) * invDij * 0.5 * stW)) * invDij;

		ApSTi += mij * gij * (pi - pj);
		ApXSPHi += mj / dj * (pi - pj) * mps::device::SPH::WKernel(dist, invHi);
	});

	ApSTi *= physParam.dt * sphMaterial.surfaceTension / mi;
	ApXSPHi *= 0.5 / physParam.dt;

	mps::device::AtomicAdd(pCGAp + id, ApSTi + ApXSPHi);
}

__global__ void ComputeCGViscosityAp_kernel(
	mps::PhysicsParam physParam,
	mps::SPHMaterialParam sphMaterial,
	const REAL3* MCUDA_RESTRICT pSPHPosition,
	const REAL* MCUDA_RESTRICT pSPHRadius,
	const REAL* MCUDA_RESTRICT pSPHMass,
	const REAL* MCUDA_RESTRICT pSPHDensity,
	size_t sphSize,
	const uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx,
	const REAL3* MCUDA_RESTRICT pCGp,
	REAL3* MCUDA_RESTRICT pCGAp)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	const auto hi = pSPHRadius[id] * mps::device::SPH::H_RATIO;
	const auto invHi = 1.0 / hi;

	const auto xi = pSPHPosition[id];
	const auto mi = pSPHMass[id];
	const auto di = pSPHDensity[id];
	const auto pi = pCGp[id];

	REAL3 Api{ 0.0 };

	mps::device::SPH::NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
	{
		const auto hj = pSPHRadius[jd] * mps::device::SPH::H_RATIO;
		const auto invHj = 1.0 / hj;
		const auto onePercentHSq = 0.01 * hi * hi;

		const auto xj = pSPHPosition[jd];
		const auto mj = pSPHMass[jd];
		const auto dj = pSPHDensity[jd];
		const auto pj = pCGp[jd];

		const auto xij = xi - xj;
		const auto pij = pi - pj;
		const auto dist = glm::length(xij);

		const auto dpij = 0.5 * (mi + mj) / dj * glm::dot(pij, xij) / (dist * dist + onePercentHSq) *
			mps::device::SPH::GKernel(dist, invHi) / (dist + FLT_EPSILON) * xij;
		Api -= dpij;
	});

	Api *= 10.0 * sphMaterial.density * sphMaterial.viscosity / di;
	mps::device::AtomicAdd(pCGAp + id, Api);
}

__global__ void ComputeFinalCGAp_kernel(
	mps::PhysicsParam physParam,
	size_t sphSize,
	const REAL3* MCUDA_RESTRICT pCGp,
	REAL3* MCUDA_RESTRICT pCGAp,
	REAL factor)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sphSize) return;

	pCGAp[id] = pCGp[id] * (1.0 + factor) + physParam.dt * pCGAp[id];
}