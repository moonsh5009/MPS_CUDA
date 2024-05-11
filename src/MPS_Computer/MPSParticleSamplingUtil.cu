#include "stdafx.h"
#include "MPSParticleSamplingUtil.cuh"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "../MPS_Object/MPSBoundaryParticleObject.h"

namespace
{
	constexpr auto nBlockSize = 256u;
}

bool mps::kernel::ParticleSampling::IsSamplingParticleRequired(const MeshMaterialParam& material, const MeshParam& obj, uint32_t* prevIdx, uint32_t* currIdx, bool* isGenerateds)
{
	if (obj.size == 0) return false;

	bool h_isApplied;
	bool* d_isApplied;
	CUDA_CHECK(cudaMalloc((void**)&d_isApplied, sizeof(bool)));
	CUDA_CHECK(cudaMemset(d_isApplied, 0, sizeof(bool)));

	ComputeSamplingNum_kernel << <mcuda::util::DivUp(obj.size, nBlockSize), nBlockSize >> > (
		material,
		obj.pFace,
		obj.pPosition,
		obj.pRTri,
		obj.pSamplingParticleSize,
		obj.pShortestEdgeID,
		obj.size,
		prevIdx,
		currIdx,
		isGenerateds,
		d_isApplied);
	CUDA_CHECK(cudaPeekAtLastError());

	CUDA_CHECK(cudaMemcpy(&h_isApplied, d_isApplied, sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_isApplied));
	return h_isApplied;
}

void mps::kernel::ParticleSampling::ParticleSampling(const MeshMaterialParam& material, const MeshParam& obj, BoundaryParticleObject& boundaryParticle)
{
	if (obj.size == 0) return;

	thrust::device_vector<uint32_t> prevIdx{ obj.size + 1u, 0u };
	thrust::device_vector<uint32_t> currIdx{ obj.size + 1u, 0u };
	thrust::device_vector<bool> isGenerateds{ obj.size, false };

	if (IsSamplingParticleRequired(material, obj, thrust::raw_pointer_cast(prevIdx.data()), thrust::raw_pointer_cast(currIdx.data()), thrust::raw_pointer_cast(isGenerateds.data())))
	{
		const auto nParticleSize = boundaryParticle.GetSize();
		thrust::device_vector<uint32_t> prevFaceID{ nParticleSize };
		thrust::device_vector<REAL2> prevBCC{ nParticleSize };
		thrust::device_vector<REAL3> prevPosition{ nParticleSize };
		thrust::device_vector<REAL> prevRadius{ nParticleSize };
		thrust::device_vector<glm::fvec4> prevColor{ nParticleSize };

		if (nParticleSize > 0u)
		{
			auto pBoundaryParticleRes = boundaryParticle.GetDeviceResource<BoundaryParticleResource>();
			if (!pBoundaryParticleRes) return;

			const auto pBoundaryParticleParam = pBoundaryParticleRes->GetBoundaryParticleParam().lock();
			if (!pBoundaryParticleParam) return;

			thrust::copy(
				thrust::device_pointer_cast(pBoundaryParticleParam->pFaceID),
				thrust::device_pointer_cast(pBoundaryParticleParam->pFaceID + nParticleSize),
				prevFaceID.begin());
			thrust::copy(
				thrust::device_pointer_cast(pBoundaryParticleParam->pBCC),
				thrust::device_pointer_cast(pBoundaryParticleParam->pBCC + nParticleSize),
				prevBCC.begin());
			thrust::copy(
				thrust::device_pointer_cast(pBoundaryParticleParam->pPosition),
				thrust::device_pointer_cast(pBoundaryParticleParam->pPosition + nParticleSize),
				prevPosition.begin());
			thrust::copy(
				thrust::device_pointer_cast(pBoundaryParticleParam->pRadius),
				thrust::device_pointer_cast(pBoundaryParticleParam->pRadius + nParticleSize),
				prevRadius.begin());
			thrust::copy(
				thrust::device_pointer_cast(pBoundaryParticleParam->pColor),
				thrust::device_pointer_cast(pBoundaryParticleParam->pColor + nParticleSize),
				prevColor.begin());
		}
		thrust::exclusive_scan(prevIdx.begin(), prevIdx.end(), prevIdx.begin());
		thrust::exclusive_scan(currIdx.begin(), currIdx.end(), currIdx.begin());

		const auto currSampNum = currIdx[obj.size];
		boundaryParticle.Resize(currSampNum);

		auto pBoundaryParticleRes = boundaryParticle.GetDeviceResource<BoundaryParticleResource>();
		if (!pBoundaryParticleRes) return;

		const auto pBoundaryParticleParam = pBoundaryParticleRes->GetBoundaryParticleParam().lock();
		if (!pBoundaryParticleParam) return;
		
		GenerateBoundaryParticle_kernel << <mcuda::util::DivUp(obj.size, nBlockSize), nBlockSize >> > (
			material,
			obj.pFace,
			obj.pPosition,
			obj.pRTri,
			obj.pShortestEdgeID,
			obj.size,
			pBoundaryParticleParam->pFaceID,
			pBoundaryParticleParam->pBCC,
			pBoundaryParticleParam->pPosition,
			pBoundaryParticleParam->pRadius,
			pBoundaryParticleParam->pColor,
			thrust::raw_pointer_cast(prevFaceID.data()),
			thrust::raw_pointer_cast(prevBCC.data()),
			thrust::raw_pointer_cast(prevPosition.data()),
			thrust::raw_pointer_cast(prevRadius.data()),
			thrust::raw_pointer_cast(prevColor.data()),
			thrust::raw_pointer_cast(prevIdx.data()),
			thrust::raw_pointer_cast(currIdx.data()),
			thrust::raw_pointer_cast(isGenerateds.data()));
		CUDA_CHECK(cudaPeekAtLastError());

		SetBarycentric_kernel << <mcuda::util::DivUp(pBoundaryParticleParam->size, nBlockSize), nBlockSize >> > (
			pBoundaryParticleParam->pFaceID,
			pBoundaryParticleParam->pBCC,
			pBoundaryParticleParam->pPosition,
			pBoundaryParticleParam->pMass,
			pBoundaryParticleParam->size,
			obj.pFace,
			obj.pPosition,
			obj.pMass);
		CUDA_CHECK(cudaPeekAtLastError());
	}
}
