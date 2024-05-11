//#pragma once
//#include "SPHUtil.h"
//#include "SPHKernel.cuh"
//
//#include "../MPS_Object/MPSUniformDef.h"
//#include "../MPS_Object/MPSSPHParam.h"
//#include "../MPS_Object/MPSBoundaryParticleParam.h"
//#include "../MPS_Object/MPSSPHMaterial.h"
//#include "../MPS_Object/MPSMeshMaterial.h"
//#include "../MPS_Object/MPSSpatialHash.h"
//#include "../MPS_Object/MPSSpatialHash.cuh"
//
//__global__ void ComputeDFSPHFactor_0_kernel(
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
//
//	REAL3 grads{ 0.0 };
//	REAL ai = 0.0;
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = sph.Position(jd);
//
//		const auto xij = xi - xj;
//		const auto dist = glm::length(xij);
//		const auto grad = material.volume * mps::device::SPH::GKernel(dist, invH) / (dist + FLT_EPSILON) * xij;
//		grads += grad;
//		ai += glm::dot(grad, grad);
//	});
//
//	sph.TempVec3(id) = grads;
//	sph.FactorDFSPH(id) = ai;
//}
//__global__ void ComputeDFSPHFactor_1_kernel(
//	mps::SPHParam sph, const mps::SPHMaterialParam sphMaterial,
//	const mps::BoundaryParticleParam boundaryParticle, const mps::MeshMaterialParam boundaryParticleMaterial,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto h = sphMaterial.radius;
//	const auto invH = 1.0 / h;
//
//	const auto xi = sph.Position(id);
//
//	REAL3 grads = sph.TempVec3(id);
//	REAL ai = sph.FactorDFSPH(id);
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = boundaryParticle.Position(jd);
//
//		const auto xij = xi - xj;
//		const auto dist = glm::length(xij);
//		const auto grad = boundaryParticle.Volume(jd) * mps::device::SPH::GKernel(dist, invH) / (dist + FLT_EPSILON) * xij;
//		grads += grad;
//		ai += glm::dot(grad, grad);
//	});
//
//	sph.TempVec3(id) = grads;
//	sph.FactorDFSPH(id) = ai;
//}
//__global__ void ComputeDFSPHFactor_2_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto grads = sph.TempVec3(id);
//	auto ai = sph.FactorDFSPH(id);
//
//	ai += glm::dot(grads, grads);
//	ai = ai > mps::device::SPH::SPH_EPSILON ? 1.0 / ai : 0.0;
//	sph.FactorDFSPH(id) = ai;
//}
//
//__global__ void ComputeDensityDelta_0_kernel(
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
//	const auto vi = sph.Velocity(id);
//
//	REAL delta = 0.0;
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = sph.Position(jd);
//		const auto vj = sph.Velocity(jd);
//
//		const auto xij = xi - xj;
//		const auto vij = vi - vj;
//		const auto dist = glm::length(xij);
//		delta += material.volume * mps::device::SPH::GKernel(dist, invH) / (dist + FLT_EPSILON) * glm::dot(xij, vij);
//	});
//
//	sph.Pressure(id) = delta;
//}
//__global__ void ComputeDensityDelta_1_kernel(
//	mps::SPHParam sph, const mps::SPHMaterialParam sphMaterial,
//	const mps::BoundaryParticleParam boundaryParticle, const mps::MeshMaterialParam boundaryParticleMaterial,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto h = sphMaterial.radius;
//	const auto invH = 1.0 / h;
//
//	const auto xi = sph.Position(id);
//	const auto vi = sph.Velocity(id);
//
//	REAL delta = sph.Pressure(id);
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = boundaryParticle.Position(jd);
//		const auto vj = boundaryParticle.Velocity(jd);
//
//		const auto xij = xi - xj;
//		const auto vij = vi - vj;
//		const auto dist = glm::length(xij);
//		delta += boundaryParticle.Volume(jd) * mps::device::SPH::GKernel(dist, invH) / (dist + FLT_EPSILON) * glm::dot(xij, vij);
//	});
//
//	sph.Pressure(id) = delta;
//}
//
//__global__ void ComputeCDStiffness_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, REAL* sumError)
//{
//	extern __shared__ REAL s_sumErrors[];
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//
//	s_sumErrors[threadIdx.x] = 0u;
//	if (id < sph.GetSize())
//	{
//		const auto di = sph.Density(id);
//		const auto delta = sph.Pressure(id);
//
//		auto stiffness = di / material.density + physParam.dt * delta - 1.0;
//		if (stiffness > 0.0)
//		{
//			s_sumErrors[threadIdx.x] = stiffness;
//			stiffness *= sph.FactorDFSPH(id) / (physParam.dt * physParam.dt);
//		}
//		else stiffness = 0.0;
//
//		sph.Pressure(id) = stiffness;
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
//__global__ void ComputeDFStiffness_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, const uint32_t* MCUDA_RESTRICT pNeiIdx, REAL* sumError)
//{
//	extern __shared__ REAL s_sumErrors[];
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//
//	s_sumErrors[threadIdx.x] = 0u;
//	if (id < sph.GetSize())
//	{
//		const auto di = sph.Density(id);
//		const auto delta = sph.Pressure(id);
//
//		//auto stiffness = mcuda::util::min(delta * physParam.dt, di / material.density + physParam.dt * delta - 0.8);
//		auto stiffness = delta * physParam.dt;
//		if (stiffness > 0.0 && pNeiIdx[id + 1] - pNeiIdx[id] > 20)
//		{
//			s_sumErrors[threadIdx.x] = stiffness;
//			stiffness *= sph.FactorDFSPH(id) / (physParam.dt * physParam.dt);
//		}
//		else stiffness = 0.0;
//
//		sph.Pressure(id) = stiffness;
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
//
//__global__ void ApplyDFSPH_0_kernel(
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
//	const auto ki = sph.Pressure(id);
//
//	REAL3 force{ 0.0 };
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = sph.Position(jd);
//		const auto kj = sph.Pressure(jd);
//
//		const auto xij = xi - xj;
//		const auto dist = glm::length(xij);
//		const auto forceij = material.volume * (ki + kj) * mps::device::SPH::GKernel(dist, invH) / (dist + FLT_EPSILON) * xij;
//		force -= forceij;
//	});
//
//	sph.Force(id) = force;
//}
//__global__ void ApplyDFSPH_1_kernel(
//	mps::SPHParam sph, const mps::SPHMaterialParam sphMaterial,
//	const mps::BoundaryParticleParam boundaryParticle, const mps::MeshMaterialParam boundaryParticleMaterial,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto h = sphMaterial.radius;
//	const auto invH = 1.0 / h;
//
//	const auto xi = sph.Position(id);
//	const auto ki = sph.Pressure(id);
//
//	REAL3 force = sph.Force(id);
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = boundaryParticle.Position(jd);
//
//		const auto xij = xi - xj;
//		const auto dist = glm::length(xij);
//		const auto forceij = boundaryParticle.Volume(jd) * ki * mps::device::SPH::GKernel(dist, invH) / (dist + FLT_EPSILON) * xij;
//		force -= forceij;
//	});
//
//	sph.Force(id) = force;
//}
//__global__ void ApplyDFSPH_2_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	auto force = sph.Force(id);
//	force = sph.Mass(id) * force;
//	sph.Force(id) = force;
//}
//
//__global__ void ApplyExplicitViscosity_kernel(
//	mps::SPHParam sph, const mps::SPHMaterialParam material,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto h = material.radius;
//	const auto invH = 1.0 / h;
//	const auto onePercentHSq = 0.01 * h * h;
//
//	const auto xi = sph.Position(id);
//	const auto vi = sph.Velocity(id);
//	const auto volumei = sph.Volume(id);
//
//	REAL3 viscForce{ 0.0 };
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = sph.Position(jd);
//		const auto xij = xi - xj;
//
//		const auto dist = glm::length(xij);
//		if (dist < h)
//		{
//			const auto vj = sph.Velocity(jd);
//			const auto volumej = sph.Volume(jd);
//
//			const auto vij = vi - vj;
//			const auto forceij = volumej * glm::dot(vij, xij) / (dist * dist + onePercentHSq) * mps::device::SPH::GKernel(dist, invH) / (dist + FLT_EPSILON) * xij;
//			viscForce += forceij;
//		}
//	});
//
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
//				xij.x * xij.x, xij.y * xij.x, xij.z * xij.x,
//				xij.x * xij.y, xij.y * xij.y, xij.z * xij.y,
//				xij.x * xij.z, xij.y * xij.z, xij.z * xij.z,
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
//				xij.x * xij.x, xij.y * xij.x, xij.z * xij.x,
//				xij.x * xij.y, xij.y * xij.y, xij.z * xij.y,
//				xij.x * xij.z, xij.y * xij.z, xij.z * xij.z,
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
//
////__global__ void ComputeJacobiViscosity_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash,
////	REAL3* vTemp, REAL omega, REAL underRelax, REAL* sumError)
////{
////	extern __shared__ REAL s_sumErrors[];
////	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
////
////	s_sumErrors[threadIdx.x] = 0u;
////	if (id < sph.GetSize())
////	{
////		const auto h = material.radius;
////		const auto invH = 1.0 / h;
////		const auto onePercentHSq = 0.01 * h * h;
////
////		const auto xi = sph.Position(id);
////
////		REAL3x3 Avi{ 0.0 };
////		REAL3 Avj{ 0.0 };
////		hash.Research(xi, [&](uint32_t jd)
////		{
////			if (id == jd) return;
////
////			const auto xj = sph.Position(jd);
////			const auto xij = xi - xj;
////
////			const auto dist = glm::length(xij);
////			if (dist < h)
////			{
////				const auto predictVj = sph.PredictVel(jd);
////				const auto volumej = sph.Volume(jd);
////				const auto Aij = volumej * mps::kernel::SPH::GKernel(dist, invH) / ((dist * dist + onePercentHSq) * dist)
////					* REAL3x3
////				{
////					xij.x* xij.x, xij.y* xij.x, xij.z* xij.x,
////						xij.x* xij.y, xij.y* xij.y, xij.z* xij.y,
////						xij.x* xij.z, xij.y* xij.z, xij.z* xij.z,
////				};
////				Avj -= predictVj * Aij;
////				Avi -= Aij;
////			}
////		});
////		const auto alpha = physParam.dt * 10.0 * material.viscosity / sph.Density(id);
////		Avj *= alpha;
////		Avi *= alpha;
////		Avi += REAL3x3{ 1.0 };
////		auto invAvi = glm::inverse(Avi);
////		if (isinf(invAvi[0][0]) || isnan(invAvi[0][0]))
////			invAvi = REAL3x3{ 0.0 };
////
////		const auto GetErrorBetweenV = [](const REAL3& v1, const REAL3& v2)
////		{
////			return mcuda::util::max(mcuda::util::max(fabs(v1.x - v2.x), fabs(v1.y - v2.y)), fabs(v1.z - v2.z));
////		};
////		const auto prevVi = sph.PreviousVel(id);
////		const auto currVi = sph.PredictVel(id);
////		auto newVi = (sph.Velocity(id) + Avj) * invAvi;
////		newVi = omega * (underRelax * (newVi - currVi) + currVi - prevVi) + prevVi;
////		s_sumErrors[threadIdx.x] = GetErrorBetweenV(newVi, sph.PredictVel(id));
////		vTemp[id] = newVi;
////	}
////	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
////	{
////		__syncthreads();
////		if (threadIdx.x < s)
////		{
////			s_sumErrors[threadIdx.x] += s_sumErrors[threadIdx.x + s];
////		}
////	}
////	__syncthreads();
////	if (threadIdx.x < 32u)
////	{
////		mcuda::util::warpSum(s_sumErrors, threadIdx.x);
////		if (threadIdx.x == 0)
////		{
////			mcuda::util::AtomicAdd(sumError, s_sumErrors[0]);
////		}
////	}
////}
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
//
//__global__ void ComputeSurfaceTensionFactor_kernel(
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
//	const auto vi = sph.Velocity(id);
//
//	REAL grad = 0.0;
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = sph.Position(jd);
//		const auto vj = sph.Velocity(jd);
//		const auto mj = sph.Mass(jd);
//
//		const auto xij = xi - xj;
//		const auto vij = vi - vj;
//		const auto dist = glm::length(xij);
//
//		const auto stG = mps::device::SPH::GKernel(dist, invH) / (dist + FLT_EPSILON) * xij;
//		const auto gradij = mj * glm::dot(vij, stG);
//		grad += gradij;
//	});
//
//	sph.FactorST(id) = grad;
//}
//
//__global__ void ComputeSurfaceTensionCGb_0_kernel(const mps::PhysicsParam physParam,
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
//	const auto vi = sph.Velocity(id);
//	const auto di = sph.Density(id);
//	const auto mi = sph.Mass(id);
//	const auto vDeltai = sph.FactorST(id);
//
//	REAL3 bi{ 0.0 };
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = sph.Position(jd);
//		const auto vj = sph.Velocity(jd);
//		const auto dj = sph.Density(jd);
//		const auto mj = sph.Mass(jd);
//		const auto vDeltaj = sph.FactorST(jd);
//
//		const auto xij = xi - xj;
//		const auto vij = vi - vj;
//		const auto dist = glm::length(xij);
//
//		const auto mij = 0.5 * (mi + mj);
//		const auto invDij = 2.0 / (di + dj);
//
//		const auto stW = mps::device::SPH::STWKernel(dist, invH);
//		const auto stG = mps::device::SPH::STGKernel(dist, invH) / (dist + FLT_EPSILON) * xij;
//		const auto gij = (stW + physParam.dt * (glm::dot(vij, stG) - (vDeltai + vDeltaj) * invDij * 0.5 * stW)) * invDij;
//
//		bi -= mij * gij * xij;
//	});
//
//	//sph.TempVec3(id) = bi;
//	sph.TempVec3(id) = material.surfaceTension / mi * physParam.dt * bi + vi;
//}
//__global__ void ComputeSurfaceTensionCGb_2_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto mi = sph.Mass(id);
//	const auto bi = sph.TempVec3(id);
//	sph.TempVec3(id) = bi * material.surfaceTension / mi * physParam.dt;
//}
//
//__global__ void ComputeSurfaceTensionCGAv_kernel(const mps::PhysicsParam physParam,
//	const mps::SPHParam sph, const mps::SPHMaterialParam material,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx,
//	const REAL3* MCUDA_RESTRICT pV,
//	REAL3* MCUDA_RESTRICT pAv)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto h = material.radius;
//	const auto invH = 1.0 / h;
//
//	const auto xi = sph.Position(id);
//	const auto vi = sph.Velocity(id);
//	const auto Vi = pV[id];
//	const auto di = sph.Density(id);
//	const auto mi = sph.Mass(id);
//	const auto vDeltai = sph.FactorST(id);
//
//	REAL3 AvST{ 0.0 };
//	REAL3 AvXSPH{ 0.0 };
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = sph.Position(jd);
//		const auto vj = sph.Velocity(jd);
//		const auto Vj = pV[jd];
//		const auto dj = sph.Density(jd);
//		const auto mj = sph.Mass(jd);
//		const auto vDeltaj = sph.FactorST(jd);
//
//		const auto xij = xi - xj;
//		const auto vij = vi - vj;
//		const auto dist = glm::length(xij);
//
//		const auto mij = 0.5 * (mi + mj);
//		const auto invDij = 2.0 / (di + dj);
//
//		const auto stW = mps::device::SPH::STWKernel(dist, invH);
//		const auto stG = mps::device::SPH::STGKernel(dist, invH) / (dist + FLT_EPSILON) * xij;
//		const auto gij = (stW + physParam.dt * (glm::dot(vij, stG) - (vDeltai + vDeltaj) * invDij * 0.5 * stW)) * invDij;
//
//		AvST += mij * gij * (Vi - Vj);
//		AvXSPH += mj / dj * (Vi - Vj) * mps::device::SPH::WKernel(dist, invH);
//	});
//
//	AvST *= physParam.dt * material.surfaceTension / mi;
//	AvXSPH *= 0.5 / physParam.dt;
//	
//	const auto Av = AvST +AvXSPH;
//	mcuda::util::AtomicAdd(reinterpret_cast<REAL*>(pAv + id), Av.x);
//	mcuda::util::AtomicAdd(reinterpret_cast<REAL*>(pAv + id) + 1u, Av.y);
//	mcuda::util::AtomicAdd(reinterpret_cast<REAL*>(pAv + id) + 2u, Av.z);
//}
//__global__ void ComputeCGViscosityAv_kernel(const mps::PhysicsParam physParam,
//	mps::SPHParam sph, const mps::SPHMaterialParam material,
//	const uint32_t* MCUDA_RESTRICT pNei,
//	const uint32_t* MCUDA_RESTRICT pNeiIdx,
//	const REAL3* MCUDA_RESTRICT pV,
//	REAL3* MCUDA_RESTRICT pAv)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto h = material.radius;
//	const auto invH = 1.0 / h;
//	const auto onePercentHSq = 0.01 * h * h;
//
//	const auto xi = sph.Position(id);
//	const auto vi = pV[id];
//	const auto mi = sph.Mass(id);
//	const auto di = sph.Density(id);
//
//	REAL3 Av{ 0.0 };
//
//	NeighborSearch(id, pNei, pNeiIdx, [&](uint32_t jd)
//	{
//		const auto xj = sph.Position(jd);
//		const auto mj = sph.Mass(jd);
//		const auto dj = sph.Density(jd);
//		const auto vj = pV[jd];
//
//		const auto xij = xi - xj;
//		const auto vij = vi - vj;
//		const auto dist = glm::length(xij);
//
//		const auto dvij = 0.5 * (mi + mj) / dj * glm::dot(vij, xij) / (dist * dist + onePercentHSq) * mps::device::SPH::GKernel(dist, invH) / (dist + FLT_EPSILON) * xij;
//		Av -= dvij;
//	});
//
//	Av *= 10.0 * material.density * material.viscosity / di;
//	mcuda::util::AtomicAdd(reinterpret_cast<REAL*>(pAv + id), Av.x);
//	mcuda::util::AtomicAdd(reinterpret_cast<REAL*>(pAv + id) + 1u, Av.y);
//	mcuda::util::AtomicAdd(reinterpret_cast<REAL*>(pAv + id) + 2u, Av.z);
//}
//__global__ void ComputeFinalCGAv_kernel(const mps::PhysicsParam physParam, const mps::SPHParam sph, const REAL3* MCUDA_RESTRICT pV, REAL3* MCUDA_RESTRICT pAv, REAL factor)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	pAv[id] = pV[id] + physParam.dt * pAv[id] + pV[id] * factor;
//}
//
//__global__ void ComputeCongugateGradientR_kernel(REAL3* MCUDA_RESTRICT pR, const REAL3* MCUDA_RESTRICT pAv, const REAL3* MCUDA_RESTRICT pB, size_t size)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= size) return;
//
//	pR[id] = pB[id] - pAv[id];
//}
//
//__global__ void ComputeCongugateGradientAlphaParam_kernel(mps::SPHParam sph, REAL3* R, REAL3* V, REAL3* Av, REAL* alpha)
//{
//	extern __shared__ REAL s_temp[];
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//
//	s_temp[threadIdx.x] = 0.0;
//	s_temp[threadIdx.x + blockDim.x] = 0.0;
//	if (id < sph.GetSize())
//	{
//		const auto ri = R[id];
//		const auto vi = V[id];
//		const auto avi = Av[id];
//		
//		s_temp[threadIdx.x] = glm::dot(ri, ri);
//		s_temp[threadIdx.x + blockDim.x] = glm::dot(vi, avi);
//	}
//#pragma unroll
//	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
//	{
//		__syncthreads();
//		if (threadIdx.x < s)
//		{
//			s_temp[threadIdx.x] += s_temp[threadIdx.x + s];
//			s_temp[threadIdx.x + blockDim.x] += s_temp[threadIdx.x + blockDim.x + s];
//		}
//	}
//	__syncthreads();
//	if (threadIdx.x < 32u)
//	{
//		mcuda::util::WarpSum(s_temp, threadIdx.x);
//		mcuda::util::WarpSum(s_temp + blockDim.x, threadIdx.x);
//		if (threadIdx.x == 0)
//		{
//			mcuda::util::AtomicAdd(alpha, s_temp[0]);
//			mcuda::util::AtomicAdd(alpha + 1, s_temp[blockDim.x]);
//		}
//	}
//}
//__global__ void UpdateCongugateGradientXR_kernel(mps::SPHParam sph, REAL3* R, REAL3* V, REAL3* Av, REAL alpha)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto ri = R[id];
//	const auto vi = V[id];
//	const auto avi = Av[id];
//	const auto predictVi = sph.PredictVel(id);
//	sph.PredictVel(id) = predictVi + alpha * vi;
//	R[id] = ri - alpha * avi;
//}
//__global__ void ComputeCongugateGradientBetaParam_kernel(mps::SPHParam sph, REAL3* R, REAL* beta)
//{
//	extern __shared__ REAL s_temp[];
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//
//	s_temp[threadIdx.x] = 0.0;
//	if (id < sph.GetSize())
//	{
//		const auto ri = R[id];
//		s_temp[threadIdx.x] = glm::dot(ri, ri);
//	}
//#pragma unroll
//	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
//	{
//		__syncthreads();
//		if (threadIdx.x < s)
//			s_temp[threadIdx.x] += s_temp[threadIdx.x + s];
//	}
//	__syncthreads();
//	if (threadIdx.x < 32u)
//	{
//		mcuda::util::WarpSum(s_temp, threadIdx.x);
//		if (threadIdx.x == 0)
//		{
//			mcuda::util::AtomicAdd(beta, s_temp[0]);
//		}
//	}
//}
//__global__ void UpdateCongugateGradientV_kernel(mps::SPHParam sph, REAL3* R, REAL3* V, REAL beta)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto ri = R[id];
//	const auto vi = V[id];
//	V[id] = ri + beta * vi;
//}
//
//__global__ void DensityColorTest_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material)
//{
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id >= sph.GetSize()) return;
//
//	const auto density = sph.Density(id);
//	const float ratio = static_cast<float>(density / material.density);
//
//	const auto blue = mcuda::util::clamp(1.0f - ratio, 0.0f, 1.0f);
//	const auto green = mcuda::util::clamp(ratio < 1.0f ? ratio : 1.0f - (ratio - 1.0f), 0.0f, 1.0f);
//	const auto red = mcuda::util::clamp(ratio - 1.0f, 0.0f, 1.0f);
//	sph.Color(id) = { red, green, blue, 1.0f };
//}