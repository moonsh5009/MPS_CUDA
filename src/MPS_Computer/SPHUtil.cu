#include "stdafx.h"
#include "SPHUtil.cuh"
#include "AdvectUtil.h"

#include <thrust/host_vector.h>
#include <thrust/extrema.h>

#define DEBUG_PRINT		0

void mps::kernel::sph::ComputeBoundaryParticleVolume_0(const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& material, const mps::SpatialHashParam& hash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = boundaryParticle.GetSize();
	if (nSize == 0) return;

	ComputeBoundaryParticleVolume_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(boundaryParticle, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}
void mps::kernel::sph::ComputeBoundaryParticleVolume_1(
	const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& material, const mps::SpatialHashParam& hash,
	const mps::BoundaryParticleParam& refBoundaryParticle, const mps::MeshMaterialParam& refMaterial, const mps::SpatialHashParam& refHash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = boundaryParticle.GetSize();
	if (nSize == 0) return;

	ComputeBoundaryParticleVolume_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(boundaryParticle, material, hash, refBoundaryParticle, refMaterial, refHash);
	CUDA_CHECK(cudaPeekAtLastError());
}
void mps::kernel::sph::ComputeBoundaryParticleVolume_2(const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& material)
{
	constexpr auto nBlockSize = 1024u;

	const auto nSize = boundaryParticle.GetSize();
	if (nSize == 0) return;

	ComputeBoundaryParticleVolume_2_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(boundaryParticle, material);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ComputeDensity_0(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeDensity_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}
void mps::kernel::sph::ComputeDensity_1(
	const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash, 
	const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeDensity_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, sphMaterial, sphHash, boundaryParticle, boundaryParticleMaterial, boundarhParticleHash);
	CUDA_CHECK(cudaPeekAtLastError());
}
void mps::kernel::sph::ComputeDensity_2(const mps::SPHParam& sph, const mps::SPHMaterialParam& material)
{
	constexpr auto nBlockSize = 1024u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeDensity_2_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ComputeDFSPHFactor_0(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeDFSPHFactor_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}
void mps::kernel::sph::ComputeDFSPHFactor_1(
	const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash,
	const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeDFSPHFactor_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, sphMaterial, sphHash, boundaryParticle, boundaryParticleMaterial, boundarhParticleHash);
	CUDA_CHECK(cudaPeekAtLastError());
}
void mps::kernel::sph::ComputeDFSPHFactor_2(const mps::SPHParam& sph, const mps::SPHMaterialParam& material)
{
	constexpr auto nBlockSize = 1024u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeDFSPHFactor_2_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ComputeDensityDelta_0(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeDensityDelta_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}
void mps::kernel::sph::ComputeDensityDelta_1(const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash, const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ComputeDensityDelta_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, sphMaterial, sphHash, boundaryParticle, boundaryParticleMaterial, boundarhParticleHash);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ApplyDFSPH_0(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ApplyDFSPH_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}
void mps::kernel::sph::ApplyDFSPH_1(const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash, const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ApplyDFSPH_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, sphMaterial, sphHash, boundaryParticle, boundaryParticleMaterial, boundarhParticleHash);
	CUDA_CHECK(cudaPeekAtLastError());
}
void mps::kernel::sph::ApplyDFSPH_2(const mps::SPHParam& sph, const mps::SPHMaterialParam& material)
{
	constexpr auto nBlockSize = 1024u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ApplyDFSPH_2_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ComputeDFSPHConstantDensity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash, const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL> d_error(1);
	thrust::host_vector<REAL> h_error(1);

	uint32_t l = 0u;
	while (l < 100u)
	{
		d_error.front() = static_cast<REAL>(0.0);

		ComputeDensityDelta_0(sph, sphMaterial, sphHash);
		ComputeDensityDelta_1(sph, sphMaterial, sphHash, boundaryParticle, boundaryParticleMaterial, boundarhParticleHash);
		ComputeCDStiffness_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> >
			(physParam, sph, sphMaterial, thrust::raw_pointer_cast(d_error.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_error = d_error;
		h_error.front() /= static_cast<REAL>(sph.GetSize() + DBL_EPSILON);
		if (h_error.front() < 1.0e-4 && l >= 2u) break;

	#if DEBUG_PRINT
		std::stringstream ss;
		ss << "Pressure Error " << l << " : " << h_error.front() << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif

		ApplyDFSPH_0(sph, sphMaterial, sphHash);
		ApplyDFSPH_1(sph, sphMaterial, sphHash, boundaryParticle, boundaryParticleMaterial, boundarhParticleHash);
		ApplyDFSPH_2(sph, sphMaterial);
		mps::kernel::UpdateVelocity(physParam, sph);
		l++;
	}
}

void mps::kernel::sph::ComputeDFSPHDivergenceFree(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash, const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL> d_error(1);
	thrust::host_vector<REAL> h_error(1);

	uint32_t l = 0u;
	while (l < 100u)
	{
		d_error.front() = static_cast<REAL>(0.0);

		ComputeDensityDelta_0(sph, sphMaterial, sphHash);
		ComputeDensityDelta_1(sph, sphMaterial, sphHash, boundaryParticle, boundaryParticleMaterial, boundarhParticleHash);
		ComputeDFStiffness_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> >
			(physParam, sph, sphMaterial, thrust::raw_pointer_cast(d_error.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_error = d_error;
		h_error.front() /= static_cast<REAL>(sph.GetSize() + DBL_EPSILON);
		if (h_error.front() < 1.0e-3 && l >= 1u) break;

	#if DEBUG_PRINT
		std::stringstream ss;
		ss << "DivergenceFree Error " << l << " : " << h_error.front() << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif

		ApplyDFSPH_0(sph, sphMaterial, sphHash);
		ApplyDFSPH_1(sph, sphMaterial, sphHash, boundaryParticle, boundaryParticleMaterial, boundarhParticleHash);
		ApplyDFSPH_2(sph, sphMaterial);
		mps::kernel::UpdateVelocity(physParam, sph);
		l++;
	}
}

void mps::kernel::sph::ApplyExplicitViscosity(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ApplyExplicitViscosity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ApplyExplicitSurfaceTension(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	ApplyExplicitSurfaceTension_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material, hash);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::sph::ApplyImplicitJacobiViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	constexpr auto nBlockSize = 256u;
	constexpr auto nApplyBlockSize = 1024u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL> d_error(3);
	thrust::host_vector<REAL> h_error(3);
	thrust::device_vector<REAL> d_omega(3, 1.0);
	thrust::host_vector<REAL> h_omega(3, 1.0);

	std::vector<thrust::device_vector<REAL3>> d_tmp(3, thrust::device_vector<REAL3>{ sph.GetSize() });

	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPredictVelArray()));
	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPreviousVelArray()));

	REAL underRelax = 0.9;
	REAL omega = 1.0;
	REAL rho = 0.995;
	constexpr REAL delta = 0.005;

	uint32_t l = 0u;
	while (l < 100u)
	{
		uint32_t minErrorID = 0u;
		if (l < 10u)
		{
			d_error[0] = static_cast<REAL>(0.0);

			ComputeJacobiViscosity_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
				physParam, sph, material, hash, thrust::raw_pointer_cast(d_tmp[0].data()));
			CUDA_CHECK(cudaPeekAtLastError());

			ComputeJacobiError_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> > (
				physParam, sph, material, hash, thrust::raw_pointer_cast(d_tmp[0].data()), thrust::raw_pointer_cast(d_error.data()));
			CUDA_CHECK(cudaPeekAtLastError());

			h_error[0] = d_error[0];
		}
		else
		{
			if (l == 10u)
			{
				h_omega[0] = 2.0 / (2.0 - rho * rho);
				h_omega[1] = 2.0 / (2.0 - (rho + delta) * (rho + delta));
				h_omega[2] = 2.0 / (2.0 - (rho - delta) * (rho - delta));
				d_omega = h_omega;
			}
			else if (l > 10u)
			{
				h_omega[0] = 4.0 / (4.0 - rho * rho * omega);
				h_omega[1] = 4.0 / (4.0 - (rho + delta) * (rho + delta) * omega);
				h_omega[2] = 4.0 / (4.0 - (rho - delta) * (rho - delta) * omega);
				d_omega = h_omega;
			}

			h_error[0] = h_error[1] = h_error[2] = static_cast<REAL>(0.0);
			d_error = h_error;

			ComputeJacobiViscosity_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
				physParam, sph, material, hash, thrust::raw_pointer_cast(d_tmp[0].data()), thrust::raw_pointer_cast(d_tmp[1].data()), thrust::raw_pointer_cast(d_tmp[2].data()),
				l, underRelax, thrust::raw_pointer_cast(d_omega.data()));
			CUDA_CHECK(cudaPeekAtLastError());

			ComputeJacobiError_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * 3 * sizeof(REAL) >> > (
				physParam, sph, material, hash, thrust::raw_pointer_cast(d_tmp[0].data()), thrust::raw_pointer_cast(d_tmp[1].data()), thrust::raw_pointer_cast(d_tmp[2].data()),
				thrust::raw_pointer_cast(d_error.data()));
			CUDA_CHECK(cudaPeekAtLastError());

			h_error = d_error;
			if (h_error[minErrorID] > h_error[1]) minErrorID = 1u;
			if (h_error[minErrorID] > h_error[2]) minErrorID = 2u;

			if (minErrorID == 1u) rho += delta;
			else if (minErrorID == 2u) rho -= delta;
			rho = std::min(std::max(0.9 + delta, rho), 1.0 - delta);
			omega = d_omega[minErrorID];
		}

		ApplyJacobiViscosity_kernel << < mcuda::util::DivUp(nSize, nApplyBlockSize), nApplyBlockSize >> >
			(sph, thrust::raw_pointer_cast(d_tmp[minErrorID].data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_error[minErrorID] *= physParam.dt / static_cast<REAL>(nSize);
		if (h_error[minErrorID] < 1.0e-4) break;


	#if DEBUG_PRINT
		std::stringstream ss;
		ss << "Implicit Jacobi Viscosity Error " << l << " : " << h_error[minErrorID];
		ss << ", Omega/MinID " << l << " : " << omega << ", " << minErrorID << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif
		l++;
	}

	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVelArray()), thrust::device_pointer_cast(sph.GetPredictVelArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
	std::stringstream ss;
	ss << "Implicit Jacobi Viscosity Loop : " << l << std::endl;
	OutputDebugStringA(ss.str().c_str());
}

//void mps::kernel::sph::ApplyImplicitJacobiViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	thrust::device_vector<REAL> d_error(1);
//	thrust::host_vector<REAL> h_error(1);
//
//	thrust::device_vector<REAL3> d_tmp(sph.GetSize());
//
//	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPredictVelArray()));
//	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPreviousVelArray()));
//
//	constexpr REAL rho = 0.9962;
//	REAL underRelax = 1.0;
//	REAL omega = 1.0;
//
//	uint32_t l = 0u;
//	while (l < 300u)
//	{
//		if (l < 10u) { omega = 1.0; underRelax = 1.0; }
//		else if (l == 10u) { omega = 2.0 / (2.0 - rho * rho); underRelax = 0.9; }
//		else { omega = 4.0 / (4.0 - rho * rho * omega); underRelax = 0.8; }
//
//		d_error.front() = static_cast<REAL>(0.0);
//
//		ComputeJacobiViscosity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> >
//			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_tmp.data()), omega, underRelax, thrust::raw_pointer_cast(d_error.data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		h_error = d_error;
//		if (h_error.front() / static_cast<REAL>(nSize) < 1.0e-1 * physParam.dt) break;
//
//	#if DEBUG_PRINT
//		std::stringstream ss;
//		ss << "Implicit Jacobi Viscosity Error " << l << " : " << h_error.front() << std::endl;
//		ss << "Omega " << l << " : " << omega << std::endl;
//		OutputDebugStringA(ss.str().c_str());
//	#endif
//
//		ApplyJacobiViscosity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
//			(sph, thrust::raw_pointer_cast(d_tmp.data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//		l++;
//	}
//
//	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVelArray()), thrust::device_pointer_cast(sph.GetPredictVelArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
//	std::stringstream ss;
//	ss << "Implicit Jacobi Viscosity Loop : " << l << std::endl;
//	OutputDebugStringA(ss.str().c_str());
//}

void mps::kernel::sph::ApplyImplicitGDViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL3> d_R(sph.GetSize());
	thrust::host_vector<REAL3> h_R(sph.GetSize());
	thrust::device_vector<REAL> d_gama(2);
	thrust::host_vector<REAL> h_gama;

	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPredictVelArray()));

	uint32_t l = 0u;
	while (l < 100u)
	{
		ComputeGDViscosityR_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_R.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		thrust::fill(d_gama.begin(), d_gama.end(), static_cast<REAL>(0.0));
		UpdateGDViscosityGama_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * 2 * sizeof(REAL) >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_gama.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_gama = d_gama;
		if (h_gama[0] < static_cast<REAL>(1.0e-2)) break;

	#if DEBUG_PRINT
		std::stringstream ss;
		ss << "Implicit GD Viscosity Error " << l << " : " << h_gama[0] << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif

		const auto gama = h_gama[0] / h_gama[1];
		UpdateGDViscosity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(sph, thrust::raw_pointer_cast(d_R.data()), gama);
		CUDA_CHECK(cudaPeekAtLastError());
		l++;
	}

	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVelArray()), thrust::device_pointer_cast(sph.GetPredictVelArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
}

void mps::kernel::sph::ApplyImplicitCGViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<REAL3> d_R(sph.GetSize());
	thrust::device_vector<REAL3> d_V(sph.GetSize());
	thrust::device_vector<REAL3> d_Av(sph.GetSize());
	thrust::device_vector<REAL> d_param(2);
	thrust::host_vector<REAL> h_param;

	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPredictVelArray()));
	ComputeGDViscosityR_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(physParam, sph, material, hash, thrust::raw_pointer_cast(d_R.data()));
	CUDA_CHECK(cudaPeekAtLastError());
	thrust::copy(d_R.begin(), d_R.end(), d_V.begin());

	constexpr REAL factor = 0.1;
	uint32_t l = 0u;
	while (l < 100u)
	{
		ComputeCGViscosityAv_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(physParam, sph, material, hash, thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), factor);
		CUDA_CHECK(cudaPeekAtLastError());

		thrust::fill(d_param.begin(), d_param.end(), static_cast<REAL>(0.0));
		UpdateCGViscosityAlphaParam_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * 2 * sizeof(REAL) >> >
			(sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), thrust::raw_pointer_cast(d_param.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_param = d_param;
		if (h_param[0] < static_cast<REAL>(1.0e-2)) break;

	#if DEBUG_PRINT
		std::stringstream ss;
		ss << "Implicit GD Viscosity Error " << l << " : " << h_param[0] << std::endl;
		OutputDebugStringA(ss.str().c_str());
	#endif

		const auto alpha = h_param[0] / h_param[1];
		UpdateCGViscosityXR_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), alpha);
		CUDA_CHECK(cudaPeekAtLastError());

		d_param[0] = static_cast<REAL>(0.0);
		UpdateCGViscosityBetaParam_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> >
			(sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_param.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		h_param[1] = d_param[0];
		const auto beta = h_param[1] / h_param[0];
		UpdateCGViscosityV_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
			(sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), beta);
		CUDA_CHECK(cudaPeekAtLastError());
		l++;
	}

	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVelArray()), thrust::device_pointer_cast(sph.GetPredictVelArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
	std::stringstream ss;
	ss << "Implicit GD Viscosity Loop : " << l << std::endl;
	OutputDebugStringA(ss.str().c_str());
}

void mps::kernel::sph::DensityColorTest(const mps::SPHParam& sph, const mps::SPHMaterialParam& material)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = sph.GetSize();
	if (nSize == 0) return;

	DensityColorTest_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(sph, material);
	CUDA_CHECK(cudaPeekAtLastError());
}