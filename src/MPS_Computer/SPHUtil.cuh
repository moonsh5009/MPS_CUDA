#pragma once
#include "SPHUtil.h"
#include "SPHKernel.cuh"

#include "../MCUDA_Lib/MCUDAHelper.cuh"
#include "../MPS_Object/MPSUniformDef.h"
#include "../MPS_Object/MPSSPHObject.h"
#include "../MPS_Object/MPSSPHMaterial.h"
#include "../MPS_Object/MPSSpatialHash.h"
#include "../MPS_Object/MPSSpatialHash.cuh"

__global__ void ComputeDensity_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;
	const auto xi = sph.Position(id);

	REAL density = material.volume * mps::kernel::sph::WKernel(0.0, invH);

	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			density += material.volume * mps::kernel::sph::WKernel(dist, invH);
		}
	});

	density *= material.density;
	sph.Density(id) = density;
}
__global__ void ComputeDFSPHFactor_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;
	const auto xi = sph.Position(id);

	REAL3 grads = { 0.0, 0.0, 0.0 };
	REAL ai = 0.0;

	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto grad = material.volume * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
			ai += glm::dot(grad, grad);
			grads += grad;
		}
	});

	ai += glm::dot(grads, grads);
	ai = ai > 1.0e-10 ? 1.0 / ai : 0.0;
	sph.FactorA(id) = ai;
}

__global__ void ComputeCDPressure_kernel(const mps::PhysicsParam physParam,
	mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash, REAL* sumError)
{
	extern __shared__ REAL s_sumErrors[];
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	s_sumErrors[threadIdx.x] = 0u;
	if (id < sph.GetSize())
	{
		const auto h = material.radius;
		const auto invH = 1.0 / h;

		const auto xi = sph.Position(id);
		const auto vi = sph.Velocity(id);
		const auto di = sph.Density(id);

		REAL delta = 0.0;
		hash.Research(xi, [&](uint32_t jd)
		{
			if (id == jd) return;

			const auto xj = sph.Position(jd);
			const auto vj = sph.Velocity(jd);
			auto xij = xi - xj;

			const auto dist = glm::length(xij);
			if (dist < h)
			{
				const auto vij = vi - vj;
				delta += material.volume * mps::kernel::sph::GKernel(dist, invH) / dist * glm::dot(xij, vij);
			}
		});

		auto pressure = di / material.density + physParam.dt * delta - 1.0;
		if (pressure > 0.0)
		{
			s_sumErrors[threadIdx.x] = pressure;
			const auto ai = sph.FactorA(id);
			pressure *= di * ai / (physParam.dt * physParam.dt);
		}
		else pressure = 0.0;
		sph.Pressure(id) = pressure;
	}
	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
	{
		__syncthreads();
		if (threadIdx.x < s)
			s_sumErrors[threadIdx.x] += s_sumErrors[threadIdx.x + s];
	}
	__syncthreads();
	if (threadIdx.x < 32u)
	{
		mcuda::util::warpSum(s_sumErrors, threadIdx.x);
		if (threadIdx.x == 0)
			mcuda::util::AtomicAdd(sumError, s_sumErrors[0]);
	}
}
__global__ void ComputeDFPressure_kernel(const mps::PhysicsParam physParam,
	mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash, REAL* sumError)
{
	extern __shared__ REAL s_sumErrors[];
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	
	s_sumErrors[threadIdx.x] = 0u;
	if (id < sph.GetSize())
	{
		const auto h = material.radius;
		const auto invH = 1.0 / h;
		const auto xi = sph.Position(id);
		const auto vi = sph.Velocity(id);
		const auto di = sph.Density(id);

		REAL delta = 0.0;
		hash.Research(xi, [&](uint32_t jd)
		{
			if (id == jd) return;

			const auto xj = sph.Position(jd);
			const auto vj = sph.Velocity(jd);
			auto xij = xi - xj;

			const auto dist = glm::length(xij);
			if (dist < h)
			{
				const auto vij = vi - vj;
				delta += material.volume * mps::kernel::sph::GKernel(dist, invH) / dist * glm::dot(xij, vij);
			}
		});

		auto pressure = mcuda::util::min(delta * physParam.dt, (di / material.density + physParam.dt * delta - 0.9));
		if (pressure > 0.0)
		{
			s_sumErrors[threadIdx.x] = pressure;
			const auto ai = sph.FactorA(id);
			pressure *= di * ai / (physParam.dt * physParam.dt);
		}
		else pressure = 0.0;
		sph.Pressure(id) = pressure;
	}
	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
	{
		__syncthreads();
		if (threadIdx.x < s)
			s_sumErrors[threadIdx.x] += s_sumErrors[threadIdx.x + s];
	}
	__syncthreads();
	if (threadIdx.x < 32u)
	{
		mcuda::util::warpSum(s_sumErrors, threadIdx.x);
		if (threadIdx.x == 0)
			mcuda::util::AtomicAdd(sumError, s_sumErrors[0]);
	}
}

__global__ void ApplyDFSPH_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;

	const auto xi = sph.Position(id);
	const auto mi = sph.Mass(id);
	const auto di = sph.Density(id);
	const auto pi = sph.Pressure(id);
	const auto ki = pi / di;

	auto force = sph.Force(id);
	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto dj = sph.Density(jd);
			const auto pj = sph.Pressure(jd);
			const auto kj = pj / dj;
			const auto forceij = (ki + kj) * mi * material.volume * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
			force -= forceij;
		}
	});
	sph.Force(id) = force;
}

__global__ void ApplyViscosity_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;
	const auto onePercentHSq = 0.01 * h * h;

	const auto xi = sph.Position(id);
	const auto vi = sph.Velocity(id);
	const auto mi = sph.Mass(id);
	const auto di = sph.Density(id);
	const auto volumei = mi / di;

	REAL3 viscForce{ 0.0 };
	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto vj = sph.Velocity(jd);
			const auto dj = sph.Density(jd);
			const auto mj = sph.Mass(jd);
			const auto volumej = mj / dj;
			const auto vij = vi - vj;
			const auto forceij = volumej * glm::dot(vij, xij) / (dist * dist + onePercentHSq) * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
			viscForce += forceij;
		}
	});
	auto force = sph.Force(id);
	force += 10.0 * material.viscosity * volumei * viscForce;
	sph.Force(id) = force;
}

__global__ void DensityColorTest_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto density = sph.Density(id);
	const float ratio = static_cast<float>(density / material.density);

	const auto blue = mcuda::util::clamp(1.0f - ratio, 0.0f, 1.0f);
	const auto green = mcuda::util::clamp(ratio < 1.0f ? ratio : 1.0f - (ratio - 1.0f), 0.0f, 1.0f);
	const auto red = mcuda::util::clamp(ratio - 1.0f, 0.0f, 1.0f);
	sph.Color(id) = { red, green, blue, 1.0f };
}