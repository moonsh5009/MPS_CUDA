#pragma once
#include "SPHUtil.h"
#include "SPHKernel.cuh"

#include "../MCUDA_Lib/MCUDAHelper.cuh"
#include "../MPS_Object/MPSUniformDef.h"
#include "../MPS_Object/MPSSPHObject.h"
#include "../MPS_Object/MPSSPHMaterial.h"
#include "../MPS_Object/MPSSpatialHash.h"
#include "../MPS_Object/MPSSpatialHash.cuh"

__global__ void ComputeColorField_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;
	const auto xi = sph.Position(id);

	REAL3 grads{ 0.0 };
	REAL det = 0.0;
	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto dj = sph.Density(jd);
			const auto volumej = sph.Mass(jd) / dj;
			det += volumej * mps::kernel::sph::WKernel(dist, invH);
			grads += volumej * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
		}
	});
	
	det *= det;
	sph.ColorField(id) = det < 1.0e-10 ? 0.0 : glm::dot(grads, grads) / det;
}

__global__ void ComputeLargeSmallDensity_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto largeH = material.radius;
	const auto smallH = material.radius * 0.4;
	const auto invLargeH = 1.0 / largeH;
	const auto invSmallH = 1.0 / smallH;

	const auto xi = sph.Position(id);
	const auto vi = sph.Velocity(id);

	REAL largeDensity = material.volume * mps::kernel::sph::WKernel(0.0, invLargeH);
	REAL smallDensity = material.volume * mps::kernel::sph::WKernel(0.0, invSmallH);

	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < largeH)
		{
			const auto vj = sph.Velocity(jd);
			const auto vij = vi - vj;

			largeDensity += material.volume * mps::kernel::sph::WKernel(dist, invLargeH);
			//largeDensity += material.volume * mps::kernel::sph::GKernel(dist, invLargeH) / dist * glm::dot(xij, vij);
			if (dist < smallH)
			{
				smallDensity += material.volume * mps::kernel::sph::WKernel(dist, invSmallH);
				//smallDensity += material.volume * mps::kernel::sph::GKernel(dist, invSmallH) / dist * glm::dot(xij, vij);
			}
		}
	});

	largeDensity *= material.density;
	smallDensity *= 0.4 * 0.4 * 0.4 * material.density;
	sph.Density(id) = largeDensity;
	sph.SmallDensity(id) = smallDensity;
}
__global__ void ComputeLargeSmallDPressure_kernel(const mps::PhysicsParam physParam,
	mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto radiusSq = material.radius * material.radius * 0.04;
	const auto largeH = material.radius;
	const auto smallH = material.radius * 0.4;

	const auto xi = sph.Position(id);
	const auto largeDi = sph.Density(id);
	const auto smallDi = sph.SmallDensity(id);

	REAL largePressure = 0.0;
	REAL smallPressure = 0.0;
	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		const auto vj = sph.Velocity(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < largeH)
		{
			const auto largeDj = sph.Density(jd);
			largePressure += (largeDj - material.density) / (4.0 * 3.141592 * material.density * physParam.dt * physParam.dt * dist);
			if (dist < smallH)
			{
				const auto smallDj = sph.SmallDensity(jd);
				smallPressure += (smallDj - material.density) * smallDj * radiusSq / (2.0 * material.density * physParam.dt * physParam.dt);
			}
		}
	});

	sph.Pressure(id) = mcuda::util::max(largePressure, 0.0);
	sph.SmallPressure(id) = mcuda::util::max(smallPressure, 0.0);
}

__global__ void ComputeSurfaceTensor_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash,
	REAL* maxColorField, REAL* maxSmallPressure)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;

	const auto largePressure = sph.Pressure(id);
	const auto smallPressure = sph.SmallPressure(id);
	const auto beta = *maxSmallPressure < 1.0e-10 ? 0.0 : 1.0 * material.surfaceTension * (*maxColorField) / (*maxSmallPressure);
	const auto pressure = mcuda::util::max(smallPressure * beta, largePressure) + material.pressureAtm;
	const auto xi = sph.Position(id);

	REAL3x3 Ci{ 0.0 };
	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);

		auto xij = xi - xj;
		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto wxij = xij* mps::kernel::sph::WKernel(0.0, invH);
			Ci += REAL3x3
			{
				wxij.x * xij.x, wxij.x * xij.y, wxij.x * xij.z,
				wxij.y * xij.x, wxij.y * xij.y, wxij.y * xij.z,
				wxij.z * xij.x, wxij.z * xij.y, wxij.z * xij.z
			};
		}
	});
	
	const auto CLength = [&Ci]()
	{
		REAL value = 0.0;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				value += Ci[i][j] * Ci[i][j];
		return sqrt(value);
	}();
	
	const auto T = CLength < 1.0e-10 ? REAL3x3{ 0.0 } : REAL3x3{ largePressure } + (pressure - largePressure) / CLength * Ci;
	sph.SurfaceTensor(id) = T;
}
__global__ void ApplySurfaceTension_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;

	const auto xi = sph.Position(id);
	const auto di = sph.Density(id);
	const auto ci = sph.ColorField(id);
	const auto Ti = sph.SurfaceTensor(id);
	const auto volumei = sph.Mass(id) / di;

	REAL3 forceColorField{ 0.0 };
	REAL3 forceAtm{ 0.0 };
	REAL3 forceInternal{ 0.0 };

	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto& xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto dj = sph.Density(jd);
			const auto cj = sph.ColorField(jd);
			const auto Tj = sph.SurfaceTensor(jd);
			const auto volumej = sph.Mass(jd) / dj;

			const auto gradij = volumej * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
			forceColorField += (ci + cj) * gradij;
			forceAtm += gradij;
			forceInternal -= (Ti + Tj) * gradij;
		}
	});

	forceColorField *= material.surfaceTension * 0.25 * volumei;
	forceAtm *= material.pressureAtm * volumei;
	forceInternal *= 0.5 * volumei;

	auto force = sph.Force(id);
	force += forceColorField;
	force += forceAtm;
	force += forceInternal;
	sph.Force(id) = force;
}