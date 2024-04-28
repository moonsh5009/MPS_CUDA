#include "stdafx.h"
#include "MPSParticleSamplingUtil.cuh"

#include <thrust/host_vector.h>
#include <thrust/extrema.h>

bool mps::kernel::ParticleSampling::IsSamplingParticleRequired(const MeshParam& obj, const MeshMaterialParam& material, uint32_t* prevIdx, uint32_t* currIdx, bool* isGenerateds)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = obj.GetSize();
	if (nSize == 0) return false;
	
	bool h_isApplied;
	bool* d_isApplied;
	CUDA_CHECK(cudaMalloc((void**)&d_isApplied, sizeof(bool)));
	CUDA_CHECK(cudaMemset(d_isApplied, 0, sizeof(bool)));

	ComputeSamplingNum_kernel << <mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(obj, material, prevIdx, currIdx, isGenerateds, d_isApplied);
	CUDA_CHECK(cudaPeekAtLastError());

	CUDA_CHECK(cudaMemcpy(&h_isApplied, d_isApplied, sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_isApplied));

	return h_isApplied;
}

void mps::kernel::ParticleSampling::ParticleSampling(const MeshParam& obj, const MeshMaterialParam& material, BoundaryParticleObject& boundaryParticle)
{
	constexpr auto nBlockSize = 256u;

	const auto nSize = obj.GetSize();
	if (nSize == 0) return;

	thrust::device_vector<uint32_t> prevIdx{ nSize + 1u, 0u };
	thrust::device_vector<uint32_t> currIdx{ nSize + 1u, 0u };
	thrust::device_vector<bool> isGenerateds{ nSize, false };

	if (IsSamplingParticleRequired(obj, material, thrust::raw_pointer_cast(prevIdx.data()), thrust::raw_pointer_cast(currIdx.data()), thrust::raw_pointer_cast(isGenerateds.data())))
	{
		const auto nParticleSize = boundaryParticle.GetSize();
		thrust::device_vector<REAL3> prevXs{ nParticleSize };
		thrust::device_vector<REAL2> prevBCC{ nParticleSize };
		thrust::device_vector<uint32_t> prevFaceID{ nParticleSize };

		if (nParticleSize > 0u)
		{
			auto pBoundaryParticleRes = boundaryParticle.GetDeviceResource<BoundaryParticleResource>();
			if (!pBoundaryParticleRes) return;

			const auto pBoundaryParticleParam = pBoundaryParticleRes->GetBoundaryParticleParam().lock();
			if (!pBoundaryParticleParam) return;

			thrust::copy(thrust::device_pointer_cast(pBoundaryParticleParam->GetPosArray()), thrust::device_pointer_cast(pBoundaryParticleParam->GetPosArray() + nParticleSize), prevXs.begin());
			thrust::copy(thrust::device_pointer_cast(pBoundaryParticleParam->GetBCCArray()), thrust::device_pointer_cast(pBoundaryParticleParam->GetBCCArray() + nParticleSize), prevBCC.begin());
			thrust::copy(thrust::device_pointer_cast(pBoundaryParticleParam->GetFaceIDArray()), thrust::device_pointer_cast(pBoundaryParticleParam->GetFaceIDArray() + nParticleSize), prevFaceID.begin());
		}
		thrust::exclusive_scan(prevIdx.begin(), prevIdx.end(), prevIdx.begin());
		thrust::exclusive_scan(currIdx.begin(), currIdx.end(), currIdx.begin());

		const auto currSampNum = currIdx[nSize];
		boundaryParticle.Resize(currSampNum);

		auto pBoundaryParticleRes = boundaryParticle.GetDeviceResource<BoundaryParticleResource>();
		if (!pBoundaryParticleRes) return;

		const auto pBoundaryParticleParam = pBoundaryParticleRes->GetBoundaryParticleParam().lock();
		if (!pBoundaryParticleParam) return;

		GenerateBoundaryParticle_kernel << <mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > 
			(obj, material, *pBoundaryParticleParam,
			thrust::raw_pointer_cast(prevXs.data()), thrust::raw_pointer_cast(prevBCC.data()), thrust::raw_pointer_cast(prevFaceID.data()),
			thrust::raw_pointer_cast(prevIdx.data()), thrust::raw_pointer_cast(currIdx.data()), thrust::raw_pointer_cast(isGenerateds.data()) );
		CUDA_CHECK(cudaPeekAtLastError());

		SetBarycentric_kernel << <mcuda::util::DivUp(currSampNum, nBlockSize), nBlockSize >> >
			(obj, material, *pBoundaryParticleParam);
		CUDA_CHECK(cudaPeekAtLastError());
	}
}
