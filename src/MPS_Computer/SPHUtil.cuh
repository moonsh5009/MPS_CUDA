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

	REAL3 grads{ 0.0 };
	REAL ai = 0.0;

	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto xj = sph.Position(jd);
			const auto grad = sph.Mass(jd) * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
			grads += grad;
			ai += glm::dot(grad, grad);
		}
	});
	ai += glm::dot(grads, grads);

	ai = ai > mps::kernel::sph::SPH_EPSILON ? 1.0 / ai : 0.0;
	sph.FactorA(id) = ai;
}
__global__ void ComputeCDStiffness_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash, REAL* sumError)
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

		auto stiffness = di / material.density + physParam.dt * delta - 1.0;
		if (stiffness > 0.0)
		{
			s_sumErrors[threadIdx.x] = stiffness;
			stiffness *= material.density * sph.FactorA(id) / (physParam.dt * physParam.dt);
		}
		else stiffness = 0.0;
		sph.Pressure(id) = stiffness;
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
__global__ void ComputeDFStiffness_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash, REAL* sumError)
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

		auto stiffness = mcuda::util::min(delta * physParam.dt, di / material.density + physParam.dt * delta - 0.8);
		//auto stiffness = delta * physParam.dt;
		if (stiffness > 0.0)
		{
			s_sumErrors[threadIdx.x] = stiffness;
			stiffness *= material.density * sph.FactorA(id) / (physParam.dt * physParam.dt);
		}
		else stiffness = 0.0;
		sph.Pressure(id) = stiffness;
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
	const auto ki = sph.Pressure(id);

	REAL3 pressureForce{ 0.0 };
	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto kj = sph.Pressure(jd);
			const auto forceij = sph.Mass(jd) * (ki + kj) * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
			pressureForce -= forceij;
		}
	});
	auto force = sph.Force(id);
	force += sph.Mass(id) * pressureForce;
	sph.Force(id) = force;
}

__global__ void ApplyExplicitViscosity_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;
	const auto onePercentHSq = 0.01 * h * h;

	const auto xi = sph.Position(id);
	const auto vi = sph.Velocity(id);
	const auto volumei = sph.Volume(id);

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
			const auto volumej = sph.Volume(jd);

			const auto vij = vi - vj;
			const auto forceij = volumej * glm::dot(vij, xij) / (dist * dist + onePercentHSq) * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
			viscForce += forceij;
		}
	});
	auto force = sph.Force(id);
	force += 10.0 * material.viscosity * volumei * viscForce;
	sph.Force(id) = force;
}
__global__ void ApplyExplicitSurfaceTension_kernel(mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;

	const auto xi = sph.Position(id);
	const auto volumei = sph.Volume(id);

	REAL3 forceST{ 0.0 };

	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto& xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto volumej = sph.Volume(jd);
			const auto forceij = volumej * mps::kernel::sph::STWKernel(dist, invH) * xij;
			forceST -= forceij;
		}
	});

	auto force = sph.Force(id);
	force += material.surfaceTension * volumei * forceST;
	sph.Force(id) = force;
}

//__global__ void ComputeJacobiViscosity_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash, 
//	REAL omega, REAL* maxError)
//{
//	extern __shared__ REAL s_maxErrors[];
//	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
//
//	s_maxErrors[threadIdx.x] = 0u;
//	if (id < sph.GetSize())
//	{
//		const auto h = material.radius;
//		const auto invH = 1.0 / h;
//		const auto onePercentHSq = 0.01 * h * h;
//
//		const auto xi = sph.Position(id);
//		const auto predictVi = sph.PredictVel(id);
//		const auto di = sph.Density(id);
//
//		REAL3 dv{ 0.0 };
//		hash.Research(xi, [&](uint32_t jd)
//		{
//			if (id == jd) return;
//
//			const auto xj = sph.Position(jd);
//			auto xij = xi - xj;
//
//			const auto dist = glm::length(xij);
//			if (dist < h)
//			{
//				const auto predictVj = sph.PreviousVel(jd);
//				const auto volumej = sph.Volume(jd);
//
//				const auto vij = predictVi - predictVj;
//				const auto dvij = volumej * glm::dot(vij, xij) / (dist * dist + onePercentHSq) * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
//				dv += dvij;
//			}
//		});
//		dv *= physParam.dt * 10.0 * material.viscosity / di;
//		const auto v0i = sph.Velocity(id);
//		const auto newVi = v0i + dv;
//		const auto prevVi = sph.PreviousVel(id);
//
//		const auto GetErrorBetweenV = [](const REAL3& v1, const REAL3& v2)
//		{
//			return mcuda::util::max(mcuda::util::max(fabs(v1.x - v2.x), fabs(v1.y - v2.y)), fabs(v1.z - v2.z));
//			//return fabs(glm::length(v1 - v2));
//		};
//
//		//const auto newPredictVi = errorRelaxation * (newVi - predictVi) + predictVi;
//		const auto newPredictVi = omega * (newVi - predictVi) + predictVi;
//		//const auto newPredictVi = omega * (errorRelaxation * (newVi - predictVi) + predictVi - prevVi) + prevVi;
//
//		s_maxErrors[threadIdx.x] = GetErrorBetweenV(newPredictVi, predictVi);
//		sph.PreviousVel(id) = predictVi;
//		sph.PredictVel(id) = newPredictVi;
//	}
//	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
//	{
//		__syncthreads();
//		if (threadIdx.x < s)
//		{
//			if (s_maxErrors[threadIdx.x] < s_maxErrors[threadIdx.x + s])
//				s_maxErrors[threadIdx.x] = s_maxErrors[threadIdx.x + s];
//		}
//	}
//	__syncthreads();
//	if (threadIdx.x < 32u)
//	{
//		mcuda::util::warpMax(s_maxErrors, threadIdx.x);
//		if (threadIdx.x == 0)
//			mcuda::util::AtomicMax(maxError, s_maxErrors[0]);
//	}
//}
__global__ void ComputeJacobiViscosity_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash,
	REAL* maxError)
{
	extern __shared__ REAL s_maxErrors[];
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	s_maxErrors[threadIdx.x] = 0u;
	if (id < sph.GetSize())
	{
		const auto h = material.radius;
		const auto invH = 1.0 / h;
		const auto onePercentHSq = 0.01 * h * h;

		const auto xi = sph.Position(id);
		const auto predictVi = sph.PredictVel(id);

		REAL3 dv{ 0.0 };
		hash.Research(xi, [&](uint32_t jd)
		{
			if (id == jd) return;

			const auto xj = sph.Position(jd);
			auto xij = xi - xj;

			const auto dist = glm::length(xij);
			if (dist < h)
			{
				const auto predictVj = sph.PredictVel(jd);
				const auto volumej = sph.Volume(jd);
				const auto vij = predictVi - predictVj;
				const auto dvij = volumej * glm::dot(vij, xij) / (dist * dist + onePercentHSq) * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
				dv += dvij;
			}
		});
		dv *= physParam.dt * 10.0 * material.viscosity / sph.Density(id);
		const auto GetErrorBetweenV = [](const REAL3& v1, const REAL3& v2)
		{
			return mcuda::util::max(mcuda::util::max(fabs(v1.x - v2.x), fabs(v1.y - v2.y)), fabs(v1.z - v2.z));
		};
		const auto newPredictVi = sph.Velocity(id) + dv;
		s_maxErrors[threadIdx.x] = GetErrorBetweenV(newPredictVi, predictVi);
		sph.PreviousVel(id) = predictVi;
		sph.PredictVel(id) = newPredictVi;
	}
	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
	{
		__syncthreads();
		if (threadIdx.x < s)
		{
			if (s_maxErrors[threadIdx.x] < s_maxErrors[threadIdx.x + s])
				s_maxErrors[threadIdx.x] = s_maxErrors[threadIdx.x + s];
		}
	}
	__syncthreads();
	if (threadIdx.x < 32u)
	{
		mcuda::util::warpMax(s_maxErrors, threadIdx.x);
		if (threadIdx.x == 0)
			mcuda::util::AtomicMax(maxError, s_maxErrors[0]);
	}
}
__global__ void ApplyJacobiViscosity_kernel(mps::SPHParam sph, REAL omega)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto prevVi = sph.PreviousVel(id);
	const auto newPredictVi = sph.PredictVel(id);
	sph.PredictVel(id) = omega * (newPredictVi - prevVi) + prevVi;
}

__global__ void ComputeGDViscosityR_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash,
	REAL3* R)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;
	const auto onePercentHSq = 0.01 * h * h;

	const auto xi = sph.Position(id);
	const auto v0i = sph.Velocity(id);
	const auto predictVi = sph.PredictVel(id);
	const auto mi = sph.Mass(id);
	const auto di = sph.Density(id);

	REAL3 dv{ 0.0 };
	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto predictVj = sph.PredictVel(jd);
			const auto mj = sph.Mass(jd);
			const auto dj = sph.Density(jd);

			const auto vij = predictVi - predictVj;
			const auto dvij = 0.5 * (mi + mj) / dj * glm::dot(vij, xij) / (dist * dist + onePercentHSq) * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
			dv += dvij;
		}
	});
	dv *= physParam.dt * 10.0 * material.viscosity / di;
	R[id] = v0i - (predictVi - dv);
}
__global__ void UpdateGDViscosityGama_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash,
	REAL3* R, REAL* gama)
{
	extern __shared__ REAL s_temp[];
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	s_temp[threadIdx.x] = 0.0;
	s_temp[threadIdx.x + blockDim.x] = 0.0;
	if (id < sph.GetSize())
	{
		const auto h = material.radius;
		const auto invH = 1.0 / h;
		const auto onePercentHSq = 0.01 * h * h;

		const auto xi = sph.Position(id);
		const auto mi = sph.Mass(id);
		const auto di = sph.Density(id);
		const auto Ri = R[id];

		REAL3 AR{ 0.0 };
		hash.Research(xi, [&](uint32_t jd)
		{
			if (id == jd) return;

			const auto xj = sph.Position(jd);
			auto xij = xi - xj;

			const auto dist = glm::length(xij);
			if (dist < h)
			{
				const auto mj = sph.Mass(jd);
				const auto dj = sph.Density(jd);
				const auto Rj = R[jd];

				const auto Rij = Ri - Rj;
				const auto dvij = 0.5 * (mi + mj) / dj * glm::dot(Rij, xij) / (dist * dist + onePercentHSq) * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
				AR += dvij;
			}
		});
		AR *= physParam.dt * 10.0 * material.viscosity / di;
		AR = Ri - AR;// +Ri * 0.1;
		s_temp[threadIdx.x] = glm::dot(Ri, Ri);
		s_temp[threadIdx.x + blockDim.x] = glm::dot(Ri, AR);
	}
	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
	{
		__syncthreads();
		if (threadIdx.x < s)
		{
			s_temp[threadIdx.x] += s_temp[threadIdx.x + s];
			s_temp[threadIdx.x + blockDim.x] += s_temp[threadIdx.x + blockDim.x + s];
		}
	}
	__syncthreads();
	if (threadIdx.x < 32u)
	{
		mcuda::util::warpSum(s_temp, threadIdx.x);
		mcuda::util::warpSum(s_temp + blockDim.x, threadIdx.x);
		if (threadIdx.x == 0)
		{
			mcuda::util::AtomicAdd(gama, s_temp[0]);
			mcuda::util::AtomicAdd(gama + 1, s_temp[blockDim.x]);
		}
	}
}
__global__ void UpdateGDViscosity_kernel(mps::SPHParam sph, REAL3* R, REAL gama)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;
	
	const auto newPredictVel = sph.PredictVel(id) + gama * R[id];
	sph.PredictVel(id) = newPredictVel;
}

__global__ void ComputeCGViscosityAv_kernel(const mps::PhysicsParam physParam, mps::SPHParam sph, const mps::SPHMaterialParam material, const mps::SpatialHashParam hash,
	REAL3* V, REAL3* Av)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto h = material.radius;
	const auto invH = 1.0 / h;
	const auto onePercentHSq = 0.01 * h * h;

	const auto xi = sph.Position(id);
	const auto vi = V[id];
	const auto mi = sph.Mass(id);
	const auto di = sph.Density(id);

	REAL3 av{ 0.0 };
	hash.Research(xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = sph.Position(jd);
		auto xij = xi - xj;

		const auto dist = glm::length(xij);
		if (dist < h)
		{
			const auto mj = sph.Mass(jd);
			const auto dj = sph.Density(jd);
			const auto vj = V[jd];

			const auto vij = vi - vj;
			const auto dvij = 0.5 * (mi + mj) / dj * glm::dot(vij, xij) / (dist * dist + onePercentHSq) * mps::kernel::sph::GKernel(dist, invH) / dist * xij;
			av += dvij;
		}
	});
	av *= physParam.dt * 10.0 * material.viscosity / di;
	av = vi - av;
	Av[id] = av;
}
__global__ void UpdateCGViscosityAlphaParam_kernel(mps::SPHParam sph, REAL3* R, REAL3* V, REAL3* Av, REAL* alpha)
{
	extern __shared__ REAL s_temp[];
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	s_temp[threadIdx.x] = 0.0;
	s_temp[threadIdx.x + blockDim.x] = 0.0;
	if (id < sph.GetSize())
	{
		const auto ri = R[id];
		const auto vi = V[id];
		const auto avi = Av[id];
		
		s_temp[threadIdx.x] = glm::dot(ri, ri);
		s_temp[threadIdx.x + blockDim.x] = glm::dot(vi, avi + vi * 0.9);
	}
	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
	{
		__syncthreads();
		if (threadIdx.x < s)
		{
			s_temp[threadIdx.x] += s_temp[threadIdx.x + s];
			s_temp[threadIdx.x + blockDim.x] += s_temp[threadIdx.x + blockDim.x + s];
		}
	}
	__syncthreads();
	if (threadIdx.x < 32u)
	{
		mcuda::util::warpSum(s_temp, threadIdx.x);
		mcuda::util::warpSum(s_temp + blockDim.x, threadIdx.x);
		if (threadIdx.x == 0)
		{
			mcuda::util::AtomicAdd(alpha, s_temp[0]);
			mcuda::util::AtomicAdd(alpha + 1, s_temp[blockDim.x]);
		}
	}
}
__global__ void UpdateCGViscosityXR_kernel(mps::SPHParam sph, REAL3* R, REAL3* V, REAL3* Av, REAL alpha)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto ri = R[id];
	const auto vi = V[id];
	const auto avi = Av[id];
	const auto predictVi = sph.PredictVel(id);
	sph.PredictVel(id) = predictVi + alpha * vi;
	R[id] = ri - alpha * avi;
}
__global__ void UpdateCGViscosityBetaParam_kernel(mps::SPHParam sph, REAL3* R, REAL* beta)
{
	extern __shared__ REAL s_temp[];
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	s_temp[threadIdx.x] = 0.0;
	if (id < sph.GetSize())
	{
		const auto ri = R[id];
		s_temp[threadIdx.x] = glm::dot(ri, ri);
	}
	for (uint32_t s = blockDim.x >> 1u; s > 32u; s >>= 1u)
	{
		__syncthreads();
		if (threadIdx.x < s)
			s_temp[threadIdx.x] += s_temp[threadIdx.x + s];
	}
	__syncthreads();
	if (threadIdx.x < 32u)
	{
		mcuda::util::warpSum(s_temp, threadIdx.x);
		if (threadIdx.x == 0)
		{
			mcuda::util::AtomicAdd(beta, s_temp[0]);
		}
	}
}
__global__ void UpdateCGViscosityV_kernel(mps::SPHParam sph, REAL3* R, REAL3* V, REAL beta)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= sph.GetSize()) return;

	const auto ri = R[id];
	const auto vi = V[id];
	V[id] = ri + beta * vi;
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