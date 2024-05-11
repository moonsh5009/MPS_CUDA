//#include "stdafx.h"
//#include "SPHUtil.cuh"
//#include "AdvectUtil.h"
//
//#include <thrust/host_vector.h>
//#include <thrust/extrema.h>
//#include "SPHUtil.h"
//
//#define DEBUG_PRINT		0
//
//void mps::kernel::SPH::ComputeBoundaryParticleVolume_0(const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = boundaryParticle.GetSize();
//	if (nSize == 0) return;
//	
//	const auto optNei = pHash->GetNeighborhood(boundaryParticle);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeBoundaryParticleVolume_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		boundaryParticle, pMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//void mps::kernel::SPH::ComputeBoundaryParticleVolume_1(
//	const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pMaterial,
//	const mps::BoundaryParticleParam& refBoundaryParticle, const mps::MeshMaterial* pRefMaterial,
//	const mps::SpatialHash* pRefHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = boundaryParticle.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pRefHash->GetNeighborhood(refBoundaryParticle);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeBoundaryParticleVolume_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		boundaryParticle, pMaterial->GetParam(), refBoundaryParticle, pRefMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//void mps::kernel::SPH::ComputeBoundaryParticleVolume_2(const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pMaterial)
//{
//	constexpr auto nBlockSize = 1024u;
//
//	const auto nSize = boundaryParticle.GetSize();
//	if (nSize == 0) return;
//
//	ComputeBoundaryParticleVolume_2_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		boundaryParticle, pMaterial->GetParam());
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//
//void mps::kernel::SPH::ComputeDensity_0(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeDensity_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//void mps::kernel::SPH::ComputeDensity_1(
//	const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
//	const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
//	const mps::SpatialHash* pSPHHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pSPHHash->GetNeighborhood(boundaryParticle);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeDensity_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pSPHMaterial->GetParam(), boundaryParticle, pBoundaryParticleMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//void mps::kernel::SPH::ComputeDensity_2(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial)
//{
//	constexpr auto nBlockSize = 1024u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	ComputeDensity_2_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pMaterial->GetParam());
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//
//void mps::kernel::SPH::ComputeDFSPHFactor_0(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeDFSPHFactor_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//void mps::kernel::SPH::ComputeDFSPHFactor_1(
//	const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
//	const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
//	const mps::SpatialHash* pSPHHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pSPHHash->GetNeighborhood(boundaryParticle);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeDFSPHFactor_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pSPHMaterial->GetParam(), boundaryParticle, pBoundaryParticleMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//void mps::kernel::SPH::ComputeDFSPHFactor_2(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial)
//{
//	constexpr auto nBlockSize = 1024u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	ComputeDFSPHFactor_2_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
//		(sph, pMaterial->GetParam());
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//
//void mps::kernel::SPH::ComputeDensityDelta_0(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeDensityDelta_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//void mps::kernel::SPH::ComputeDensityDelta_1(
//	const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
//	const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
//	const mps::SpatialHash* pSPHHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pSPHHash->GetNeighborhood(boundaryParticle);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeDensityDelta_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pSPHMaterial->GetParam(), boundaryParticle, pBoundaryParticleMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//
//void mps::kernel::SPH::ApplyDFSPH_0(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ApplyDFSPH_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//void mps::kernel::SPH::ApplyDFSPH_1(
//	const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
//	const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
//	const mps::SpatialHash* pSPHHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pSPHHash->GetNeighborhood(boundaryParticle);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ApplyDFSPH_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pSPHMaterial->GetParam(), boundaryParticle, pBoundaryParticleMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//void mps::kernel::SPH::ApplyDFSPH_2(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial)
//{
//	constexpr auto nBlockSize = 1024u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	ApplyDFSPH_2_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pMaterial->GetParam());
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//
//void mps::kernel::SPH::ComputeDFSPHConstantDensity(const mps::PhysicsParam& physParam,
//	const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
//	const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
//	const mps::SpatialHash* pSPHHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	thrust::device_vector<REAL> d_error(1);
//	thrust::host_vector<REAL> h_error(1);
//
//	uint32_t l = 0u;
//	while (l < 100u)
//	{
//		d_error.front() = static_cast<REAL>(0.0);
//
//		ComputeDensityDelta_0(sph, pSPHMaterial, pSPHHash);
//		ComputeDensityDelta_1(sph, pSPHMaterial, boundaryParticle, pBoundaryParticleMaterial, pSPHHash);
//		ComputeCDStiffness_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> > (
//			physParam, sph, pSPHMaterial->GetParam(), thrust::raw_pointer_cast(d_error.data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		h_error = d_error;
//		h_error.front() /= static_cast<REAL>(sph.GetSize() + DBL_EPSILON);
//		if (h_error.front() < 1.0e-4 && l >= 2u) break;
//
//	#if DEBUG_PRINT
//		std::stringstream ss;
//		ss << "Pressure Error " << l << " : " << h_error.front() << std::endl;
//		OutputDebugStringA(ss.str().c_str());
//	#endif
//
//		ApplyDFSPH_0(sph, pSPHMaterial, pSPHHash);
//		ApplyDFSPH_1(sph, pSPHMaterial, boundaryParticle, pBoundaryParticleMaterial, pSPHHash);
//		ApplyDFSPH_2(sph, pSPHMaterial);
//		mps::kernel::UpdateVelocity(physParam, sph);
//		l++;
//	}
//}
//void mps::kernel::SPH::ComputeDFSPHDivergenceFree(const mps::PhysicsParam& physParam,
//	const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
//	const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
//	const mps::SpatialHash* pSPHHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pSPHHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	thrust::device_vector<REAL> d_error(1);
//	thrust::host_vector<REAL> h_error(1);
//
//	uint32_t l = 0u;
//	while (l < 100u)
//	{
//		d_error.front() = static_cast<REAL>(0.0);
//
//		ComputeDensityDelta_0(sph, pSPHMaterial, pSPHHash);
//		ComputeDensityDelta_1(sph, pSPHMaterial, boundaryParticle, pBoundaryParticleMaterial, pSPHHash);
//		ComputeDFStiffness_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> > (
//			physParam, sph, pSPHMaterial->GetParam(), pNeiIdx, thrust::raw_pointer_cast(d_error.data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		h_error = d_error;
//		h_error.front() /= static_cast<REAL>(sph.GetSize() + DBL_EPSILON);
//		if (h_error.front() < 1.0e-3 && l >= 1u) break;
//
//	#if DEBUG_PRINT
//		std::stringstream ss;
//		ss << "DivergenceFree Error " << l << " : " << h_error.front() << std::endl;
//		OutputDebugStringA(ss.str().c_str());
//	#endif
//
//		ApplyDFSPH_0(sph, pSPHMaterial, pSPHHash);
//		ApplyDFSPH_1(sph, pSPHMaterial, boundaryParticle, pBoundaryParticleMaterial, pSPHHash);
//		ApplyDFSPH_2(sph, pSPHMaterial);
//		mps::kernel::UpdateVelocity(physParam, sph);
//		l++;
//	}
//}
//
//void mps::kernel::SPH::ApplyExplicitViscosity(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ApplyExplicitViscosity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//
//void mps::kernel::SPH::ApplyExplicitSurfaceTension(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ApplyExplicitSurfaceTension_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//
//void mps::kernel::SPH::ApplyImplicitJacobiViscosity(const mps::PhysicsParam& physParam,
//	const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//	constexpr auto nApplyBlockSize = 1024u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	thrust::device_vector<REAL> d_error(3);
//	thrust::host_vector<REAL> h_error(3);
//	thrust::device_vector<REAL> d_omega(3, 1.0);
//	thrust::host_vector<REAL> h_omega(3, 1.0);
//
//	std::vector<thrust::device_vector<REAL3>> d_tmp(3, thrust::device_vector<REAL3>{ sph.GetSize() });
//
//	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPredictVelArray()));
//	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPreviousVelArray()));
//
//	REAL underRelax = 0.9;
//	REAL omega = 1.0;
//	REAL rho = 0.995;
//	constexpr REAL delta = 0.005;
//
//	uint32_t l = 0u;
//	while (l < 100u)
//	{
//		uint32_t minErrorID = 0u;
//		if (l < 10u)
//		{
//			d_error[0] = static_cast<REAL>(0.0);
//
//			ComputeJacobiViscosity_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//				physParam, sph, pMaterial->GetParam(), pNei, pNeiIdx, thrust::raw_pointer_cast(d_tmp[0].data()));
//			CUDA_CHECK(cudaPeekAtLastError());
//
//			ComputeJacobiError_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> > (
//				physParam, sph, pMaterial->GetParam(), pNei, pNeiIdx, thrust::raw_pointer_cast(d_tmp[0].data()), thrust::raw_pointer_cast(d_error.data()));
//			CUDA_CHECK(cudaPeekAtLastError());
//
//			h_error[0] = d_error[0];
//		}
//		else
//		{
//			if (l == 10u)
//			{
//				h_omega[0] = 2.0 / (2.0 - rho * rho);
//				h_omega[1] = 2.0 / (2.0 - (rho + delta) * (rho + delta));
//				h_omega[2] = 2.0 / (2.0 - (rho - delta) * (rho - delta));
//				d_omega = h_omega;
//			}
//			else if (l > 10u)
//			{
//				h_omega[0] = 4.0 / (4.0 - rho * rho * omega);
//				h_omega[1] = 4.0 / (4.0 - (rho + delta) * (rho + delta) * omega);
//				h_omega[2] = 4.0 / (4.0 - (rho - delta) * (rho - delta) * omega);
//				d_omega = h_omega;
//			}
//
//			h_error[0] = h_error[1] = h_error[2] = static_cast<REAL>(0.0);
//			d_error = h_error;
//
//			ComputeJacobiViscosity_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//				physParam, sph, pMaterial->GetParam(), pNei, pNeiIdx,
//				thrust::raw_pointer_cast(d_tmp[0].data()), thrust::raw_pointer_cast(d_tmp[1].data()), thrust::raw_pointer_cast(d_tmp[2].data()),
//				l, underRelax, thrust::raw_pointer_cast(d_omega.data()));
//			CUDA_CHECK(cudaPeekAtLastError());
//
//			ComputeJacobiError_1_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * 3 * sizeof(REAL) >> > (
//				physParam, sph, pMaterial->GetParam(), pNei, pNeiIdx,
//				thrust::raw_pointer_cast(d_tmp[0].data()), thrust::raw_pointer_cast(d_tmp[1].data()), thrust::raw_pointer_cast(d_tmp[2].data()),
//				thrust::raw_pointer_cast(d_error.data()));
//			CUDA_CHECK(cudaPeekAtLastError());
//
//			h_error = d_error;
//			if (h_error[minErrorID] > h_error[1]) minErrorID = 1u;
//			if (h_error[minErrorID] > h_error[2]) minErrorID = 2u;
//
//			if (minErrorID == 1u) rho += delta;
//			else if (minErrorID == 2u) rho -= delta;
//			rho = std::min(std::max(0.5 + delta, rho), 1.0 - delta);
//			omega = d_omega[minErrorID];
//		}
//
//		ApplyJacobiViscosity_kernel << < mcuda::util::DivUp(nSize, nApplyBlockSize), nApplyBlockSize >> > (
//			sph, thrust::raw_pointer_cast(d_tmp[minErrorID].data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		h_error[minErrorID] *= physParam.dt / static_cast<REAL>(nSize);
//		if (h_error[minErrorID] < 1.0e-4) break;
//
//
//	#if DEBUG_PRINT
//		std::stringstream ss;
//		ss << "Implicit Jacobi Viscosity Error " << l << " : " << h_error[minErrorID];
//		ss << ", Omega/MinID " << l << " : " << omega << ", " << minErrorID << std::endl;
//		OutputDebugStringA(ss.str().c_str());
//	#endif
//		l++;
//	}
//
//	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVelArray()), thrust::device_pointer_cast(sph.GetPredictVelArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
//	std::stringstream ss;
//	ss << "Implicit Jacobi Viscosity Loop : " << l << std::endl;
//	OutputDebugStringA(ss.str().c_str());
//}
//
//void mps::kernel::SPH::ApplyImplicitCGViscosity(const mps::PhysicsParam& physParam,
//	const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	/*constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	thrust::device_vector<REAL3> d_R(sph.GetSize());
//	thrust::device_vector<REAL3> d_V(sph.GetSize());
//	thrust::device_vector<REAL3> d_Av(sph.GetSize());
//	thrust::device_vector<REAL> d_param(2);
//	thrust::host_vector<REAL> h_param;
//
//	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetPredictVelArray()));
//	ComputeGDViscosityR_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		physParam, sph, pMaterial->GetParam(), pNei, pNeiIdx, thrust::raw_pointer_cast(d_R.data()));
//	CUDA_CHECK(cudaPeekAtLastError());
//	thrust::copy(d_R.begin(), d_R.end(), d_V.begin());
//
//	REAL factor = 0.01;
//	uint32_t l = 0u;
//	while (l < 100u)
//	{
//		if (l < 18u)		factor = 0.01;
//		else if (l == 18u)	factor = 0.1;
//		else				factor += l * 0.016;
//
//		ComputeCGViscosityAv_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//			physParam, sph, pMaterial->GetParam(), pNei, pNeiIdx, thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), factor);
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		thrust::fill(d_param.begin(), d_param.end(), static_cast<REAL>(0.0));
//		UpdateCGViscosityAlphaParam_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * 2 * sizeof(REAL) >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), thrust::raw_pointer_cast(d_param.data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		h_param = d_param;
//		if (h_param[0] < static_cast<REAL>(1.0e-2)) break;
//
//	#if DEBUG_PRINT
//		std::stringstream ss;
//		ss << "Implicit GD Viscosity Error " << l << " : " << h_param[0] << std::endl;
//		OutputDebugStringA(ss.str().c_str());
//	#endif
//
//		const auto alpha = h_param[0] / h_param[1];
//		UpdateCGViscosityXR_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), alpha);
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		d_param[0] = static_cast<REAL>(0.0);
//		UpdateCGViscosityBetaParam_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_param.data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		h_param[1] = d_param[0];
//		const auto beta = h_param[1] / h_param[0];
//		UpdateCGViscosityV_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), beta);
//		CUDA_CHECK(cudaPeekAtLastError());
//		l++;
//	}
//
//	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVelArray()), thrust::device_pointer_cast(sph.GetPredictVelArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
//	std::stringstream ss;
//	ss << "Implicit GD Viscosity Loop : " << l << std::endl;
//	OutputDebugStringA(ss.str().c_str());*/
//}
//
//void mps::kernel::SPH::ComputeSurfaceTensionFactor(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeSurfaceTensionFactor_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//
//void mps::kernel::SPH::ComputeSurfaceTensionCGb(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize01 = 256u;
//	constexpr auto nBlockSize2 = 1024u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeSurfaceTensionCGb_0_kernel << < mcuda::util::DivUp(nSize, nBlockSize01), nBlockSize01 >> > (
//		physParam, sph, pMaterial->GetParam(), pNei, pNeiIdx);
//	CUDA_CHECK(cudaPeekAtLastError());
//	/*ComputeSurfaceTensionCGb_2_kernel << < mcuda::util::DivUp(nSize, nBlockSize2), nBlockSize2 >> > (
//		physParam, sph, pMaterial->GetParam());
//	CUDA_CHECK(cudaPeekAtLastError());*/
//}
//
//void mps::kernel::SPH::ComputeSurfaceTensionCGAv(const mps::PhysicsParam& physParam,
//	const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash, REAL3* pV, REAL3* pAv)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeSurfaceTensionCGAv_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		physParam, sph, pMaterial->GetParam(), pNei, pNeiIdx, pV, pAv);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//
//void mps::kernel::SPH::ApplyImplicitCGSurfaceTension(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//	constexpr auto nMaxBlockSize = 1024u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	thrust::device_vector<REAL3> d_R(nSize);
//	thrust::device_vector<REAL3> d_V(nSize);
//	thrust::device_vector<REAL3> d_Av(nSize);
//	thrust::device_vector<REAL> d_param(2);
//	thrust::host_vector<REAL> h_param;
//
//	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + nSize), thrust::device_pointer_cast(sph.GetPredictVelArray()));
//	
//	ComputeSurfaceTensionFactor(sph, pMaterial, pHash);
//	ComputeSurfaceTensionCGb(physParam, sph, pMaterial, pHash);
//
//	thrust::fill(d_Av.begin(), d_Av.end(), REAL3{ 0.0 });
//	ComputeSurfaceTensionCGAv(physParam, sph, pMaterial, pHash, sph.GetPredictVelArray(), thrust::raw_pointer_cast(d_Av.data()));
//	ComputeFinalCGAv_kernel << < mcuda::util::DivUp(nSize, nMaxBlockSize), nMaxBlockSize >> > (
//		physParam, sph, sph.GetPredictVelArray(), thrust::raw_pointer_cast(d_Av.data()), 0.0);
//	CUDA_CHECK(cudaPeekAtLastError());
//	ComputeCongugateGradientR_kernel << < mcuda::util::DivUp(nSize, nMaxBlockSize), nMaxBlockSize >> > (
//		thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_Av.data()), sph.GetTempVec3Array(), nSize);
//	CUDA_CHECK(cudaPeekAtLastError());
//
//	thrust::copy(d_R.begin(), d_R.end(), d_V.begin());
//
//	REAL factor = 0.0;
//	uint32_t l = 0u;
//	while (l < 100u)
//	{
//		/*if (l < 18u)		factor = 0.01;
//		else if (l == 18u)	factor = 0.1;
//		else				factor += l * 0.016;*/
//
//		thrust::fill(d_Av.begin(), d_Av.end(), REAL3{ 0.0 });
//		ComputeSurfaceTensionCGAv(physParam, sph, pMaterial, pHash, thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()));
//		ComputeFinalCGAv_kernel << < mcuda::util::DivUp(nSize, nMaxBlockSize), nMaxBlockSize >> > (
//			physParam, sph, thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), factor);
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		thrust::fill(d_param.begin(), d_param.end(), static_cast<REAL>(0.0));
//		ComputeCongugateGradientAlphaParam_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * 2 * sizeof(REAL) >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), thrust::raw_pointer_cast(d_param.data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		h_param = d_param;
//		if (h_param[0] < static_cast<REAL>(1.0e-2)) break;
//
//	#if DEBUG_PRINT
//		std::stringstream ss;
//		ss << "Implicit GD SurfaceTension Error " << l << " : " << h_param[0] << ", " << h_param[1] << std::endl;
//		OutputDebugStringA(ss.str().c_str());
//	#endif
//
//		const auto alpha = h_param[0] / h_param[1];
//		UpdateCongugateGradientXR_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), alpha);
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		d_param[0] = static_cast<REAL>(0.0);
//		ComputeCongugateGradientBetaParam_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_param.data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		h_param[1] = d_param[0];
//		const auto beta = h_param[1] / h_param[0];
//		UpdateCongugateGradientV_kernel << < mcuda::util::DivUp(nSize, nMaxBlockSize), nMaxBlockSize >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), beta);
//		CUDA_CHECK(cudaPeekAtLastError());
//		l++;
//	}
//
//	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVelArray()), thrust::device_pointer_cast(sph.GetPredictVelArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
//	std::stringstream ss;
//	ss << "Implicit GD SurfaceTension Loop : " << l << std::endl;
//	OutputDebugStringA(ss.str().c_str());
//}
//
//void mps::kernel::SPH::ComputeViscosityCGAv(const mps::PhysicsParam& physParam,
//	const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash, REAL3* pV, REAL3* pAv)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	ComputeCGViscosityAv_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		physParam, sph, pMaterial->GetParam(), pNei, pNeiIdx, pV, pAv);
//	CUDA_CHECK(cudaPeekAtLastError());
//}
//
//void mps::kernel::SPH::ApplyImplicitViscosityNSurfaceTension(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash)
//{
//	constexpr auto nBlockSize = 256u;
//	constexpr auto nMaxBlockSize = 1024u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	const auto optNei = pHash->GetNeighborhood(sph);
//	if (!optNei) return;
//	const auto& [pNei, pNeiIdx] = optNei.value();
//
//	thrust::device_vector<REAL3> d_R(nSize);
//	thrust::device_vector<REAL3> d_V(nSize);
//	thrust::device_vector<REAL3> d_Av(nSize);
//	thrust::device_vector<REAL> d_param(2);
//	thrust::host_vector<REAL> h_param;
//
//	thrust::copy(thrust::device_pointer_cast(sph.GetVelocityArray()), thrust::device_pointer_cast(sph.GetVelocityArray() + nSize), thrust::device_pointer_cast(sph.GetPredictVelArray()));
//
//	ComputeSurfaceTensionFactor(sph, pMaterial, pHash);
//	ComputeSurfaceTensionCGb(physParam, sph, pMaterial, pHash);
//
//	thrust::fill(d_Av.begin(), d_Av.end(), REAL3{ 0.0 });
//	ComputeSurfaceTensionCGAv(physParam, sph, pMaterial, pHash, sph.GetPredictVelArray(), thrust::raw_pointer_cast(d_Av.data()));
//	ComputeViscosityCGAv(physParam, sph, pMaterial, pHash, sph.GetPredictVelArray(), thrust::raw_pointer_cast(d_Av.data()));
//	ComputeFinalCGAv_kernel << < mcuda::util::DivUp(nSize, nMaxBlockSize), nMaxBlockSize >> > (
//		physParam, sph, sph.GetPredictVelArray(), thrust::raw_pointer_cast(d_Av.data()), 0.0);
//	CUDA_CHECK(cudaPeekAtLastError());
//
//	ComputeCongugateGradientR_kernel << < mcuda::util::DivUp(nSize, nMaxBlockSize), nMaxBlockSize >> > (
//		thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_Av.data()), sph.GetTempVec3Array(), nSize);
//	CUDA_CHECK(cudaPeekAtLastError());
//
//	thrust::copy(d_R.begin(), d_R.end(), d_V.begin());
//
//	REAL factor = 0.01;
//	uint32_t l = 0u;
//	while (l < 100u)
//	{
//		/*if (l < 18u)		factor = 0.01;
//		else if (l == 18u)	factor = 0.1;
//		else				factor += l * 0.016;*/
//
//		thrust::fill(d_Av.begin(), d_Av.end(), REAL3{ 0.0 });
//		ComputeSurfaceTensionCGAv(physParam, sph, pMaterial, pHash, thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()));
//		ComputeViscosityCGAv(physParam, sph, pMaterial, pHash, thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()));
//		ComputeFinalCGAv_kernel << < mcuda::util::DivUp(nSize, nMaxBlockSize), nMaxBlockSize >> > (
//			physParam, sph, thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), factor);
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		thrust::fill(d_param.begin(), d_param.end(), static_cast<REAL>(0.0));
//		ComputeCongugateGradientAlphaParam_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * 2 * sizeof(REAL) >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), thrust::raw_pointer_cast(d_param.data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		h_param = d_param;
//		if (h_param[0] < static_cast<REAL>(1.0e-2)) break;
//		  
//	//#if DEBUG_PRINT
//		std::stringstream ss;
//		ss << "Implicit GD SurfaceTension Error " << l << " : " << h_param[0] << ", " << h_param[1] << std::endl;
//		OutputDebugStringA(ss.str().c_str());
//	//#endif
//
//		const auto alpha = h_param[0] / h_param[1];
//		UpdateCongugateGradientXR_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), thrust::raw_pointer_cast(d_Av.data()), alpha);
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		d_param[0] = static_cast<REAL>(0.0);
//		ComputeCongugateGradientBetaParam_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, nBlockSize * sizeof(REAL) >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_param.data()));
//		CUDA_CHECK(cudaPeekAtLastError());
//
//		h_param[1] = d_param[0];
//		const auto beta = h_param[1] / h_param[0];
//		UpdateCongugateGradientV_kernel << < mcuda::util::DivUp(nSize, nMaxBlockSize), nMaxBlockSize >> > (
//			sph, thrust::raw_pointer_cast(d_R.data()), thrust::raw_pointer_cast(d_V.data()), beta);
//		CUDA_CHECK(cudaPeekAtLastError());
//		l++;
//	}
//
//	thrust::copy(thrust::device_pointer_cast(sph.GetPredictVelArray()), thrust::device_pointer_cast(sph.GetPredictVelArray() + sph.GetSize()), thrust::device_pointer_cast(sph.GetVelocityArray()));
//	std::stringstream ss;
//	ss << "Implicit GD SurfaceTension Loop : " << l << std::endl;
//	OutputDebugStringA(ss.str().c_str());
//}
//
//void mps::kernel::SPH::DensityColorTest(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial)
//{
//	constexpr auto nBlockSize = 256u;
//
//	const auto nSize = sph.GetSize();
//	if (nSize == 0) return;
//
//	DensityColorTest_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
//		sph, pMaterial->GetParam());
//	CUDA_CHECK(cudaPeekAtLastError());
//}