#include "stdafx.h"
#include "MPSSpatialHash.cuh"
#include <thrust/sort.h>

#include "MPSSPHParam.h"
#include "MPSBoundaryParticleParam.h"

namespace
{
	constexpr auto nFullBlockSize = 1024u;
	constexpr auto nSearchBlockSize = 256u;
}

mps::SpatialHash::SpatialHash() : mps::VirtualTree<SpatialHashParam>{}
{
}

void mps::SpatialHash::SetObjectSize(size_t size)
{
	m_key.resize(size);
	m_ID.resize(size);

	GetParam().objSize = size;
	GetParam().pKey = thrust::raw_pointer_cast(m_key.data());
	GetParam().pID = thrust::raw_pointer_cast(m_ID.data());
}

void mps::SpatialHash::SetCeilSize(REAL size)
{
	GetParam().ceilSize = size;
}

void mps::SpatialHash::SetHashSize(const glm::uvec3& size)
{
	const auto hashCeilSize = size.x * size.y * size.z;
	m_startIdx.resize(hashCeilSize);
	m_endIdx.resize(hashCeilSize);

	GetParam().hashSize = size;
	GetParam().pStartIdx = thrust::raw_pointer_cast(m_startIdx.data());
	GetParam().pEndIdx = thrust::raw_pointer_cast(m_endIdx.data());
}

std::optional<mps::NeiParam> mps::SpatialHash::GetNeighborhood(const ParticleParam& objParam) const
{
	const auto iter = m_mapNei.find(objParam.phaseID);
	if (iter == m_mapNei.end()) return {};

	const auto& [nei, neiIdx] = iter->second;
	return mps::NeiParam{ thrust::raw_pointer_cast(nei.data()), thrust::raw_pointer_cast(neiIdx.data()) };
}

void mps::SpatialHash::UpdateHash(const mps::ParticleParam& particleParam)
{
	if (particleParam.size == 0) return;

	thrust::fill(m_startIdx.begin(), m_startIdx.end(), 0xffffffff);
	thrust::fill(m_endIdx.begin(), m_endIdx.end(), 0xffffffff);

	InitHash_kernel << < mcuda::util::DivUp(particleParam.size, nFullBlockSize), nFullBlockSize >> > (
		particleParam.pPosition,
		particleParam.size,
		GetParam().pKey,
		GetParam().pID,
		GetParam().hashSize,
		GetParam().ceilSize);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort_by_key(m_key.begin(), m_key.end(), m_ID.begin());

	ReorderHash_kernel << <mcuda::util::DivUp(particleParam.size, nFullBlockSize), nFullBlockSize, (nFullBlockSize + 1) * sizeof(uint32_t) >> > (
		particleParam.size,
		GetParam().pKey,
		GetParam().pStartIdx,
		GetParam().pEndIdx);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::SpatialHash::ZSort(mps::ParticleParam& particleParam)
{
	if (particleParam.size == 0) return;

	thrust::fill(m_startIdx.begin(), m_startIdx.end(), 0xffffffff);
	thrust::fill(m_endIdx.begin(), m_endIdx.end(), 0xffffffff);

	InitHashZIndex_kernel << < mcuda::util::DivUp(particleParam.size, nFullBlockSize), nFullBlockSize >> > (
		particleParam.pPosition,
		particleParam.size,
		GetParam().pKey,
		GetParam().hashSize,
		GetParam().ceilSize);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort_by_key(m_key.begin(), m_key.end(), thrust::make_zip_iterator(
		thrust::device_pointer_cast(particleParam.pPosition),
		thrust::device_pointer_cast(particleParam.pMass),
		thrust::device_pointer_cast(particleParam.pVelocity),
		thrust::device_pointer_cast(particleParam.pColor)));
}

void mps::SpatialHash::ZSort(mps::SPHParam& sphParam)
{
	if (sphParam.size == 0) return;

	thrust::fill(m_startIdx.begin(), m_startIdx.end(), 0xffffffff);
	thrust::fill(m_endIdx.begin(), m_endIdx.end(), 0xffffffff);

	InitHashZIndex_kernel << < mcuda::util::DivUp(sphParam.size, nFullBlockSize), nFullBlockSize >> > (
		sphParam.pPosition,
		sphParam.size,
		GetParam().pKey,
		GetParam().hashSize,
		GetParam().ceilSize);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort_by_key(m_key.begin(), m_key.end(), thrust::make_zip_iterator(
		thrust::device_pointer_cast(sphParam.pPosition),
		thrust::device_pointer_cast(sphParam.pMass),
		thrust::device_pointer_cast(sphParam.pVelocity),
		thrust::device_pointer_cast(sphParam.pColor),
		thrust::device_pointer_cast(sphParam.pRadius)));
}

void mps::SpatialHash::ZSort(mps::BoundaryParticleParam& boundaryParticleParam)
{
	if (boundaryParticleParam.size == 0) return;

	InitHashZIndex_kernel << < mcuda::util::DivUp(boundaryParticleParam.size, nFullBlockSize), nFullBlockSize >> > (
		boundaryParticleParam.pPosition, boundaryParticleParam.size, GetParam().pKey, GetParam().hashSize, GetParam().ceilSize);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort_by_key(m_key.begin(), m_key.end(), thrust::make_zip_iterator(
		thrust::device_pointer_cast(boundaryParticleParam.pPosition),
		thrust::device_pointer_cast(boundaryParticleParam.pMass),
		thrust::device_pointer_cast(boundaryParticleParam.pVelocity),
		thrust::device_pointer_cast(boundaryParticleParam.pColor),
		thrust::device_pointer_cast(boundaryParticleParam.pRadius),
		thrust::device_pointer_cast(boundaryParticleParam.pFaceID),
		thrust::device_pointer_cast(boundaryParticleParam.pBCC)));
}

void mps::SpatialHash::BuildNeighorhood(const mps::ParticleParam& particleParam)
{
	if (particleParam.size == 0) return;

	auto& [nei, neiIdx] = m_mapNei.emplace(particleParam.phaseID,
		std::tuple<thrust::device_vector<uint32_t>, thrust::device_vector<uint32_t>>{}).first->second;
	neiIdx.resize(particleParam.size + 1u);

	ComputeNeighborhoodSize_kernel << < mcuda::util::DivUp(particleParam.size, nSearchBlockSize), nSearchBlockSize >> > (
		particleParam.pPosition,
		particleParam.pRadius,
		particleParam.size,
		GetParam().pStartIdx,
		GetParam().pEndIdx,
		GetParam().pID,
		GetParam().hashSize,
		GetParam().ceilSize,
		thrust::raw_pointer_cast(neiIdx.data()));
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::inclusive_scan(neiIdx.begin(), neiIdx.end(), neiIdx.begin());
	nei.resize(neiIdx.back());

	BuildNeighborhood_kernel << < mcuda::util::DivUp(particleParam.size, nSearchBlockSize), nSearchBlockSize >> > (
		particleParam.pPosition,
		particleParam.pRadius,
		particleParam.size,
		GetParam().pStartIdx,
		GetParam().pEndIdx,
		GetParam().pID,
		GetParam().hashSize,
		GetParam().ceilSize,
		thrust::raw_pointer_cast(nei.data()),
		thrust::raw_pointer_cast(neiIdx.data()));
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::SpatialHash::BuildNeighorhood(const mps::ParticleParam& particleParam, const mps::ParticleParam& refParticleParam, const SpatialHash* pRefHash)
{
	if (particleParam.size == 0 || refParticleParam.size == 0) return;

	auto& [nei, neiIdx] = m_mapNei.emplace(refParticleParam.phaseID,
		std::tuple<thrust::device_vector<uint32_t>, thrust::device_vector<uint32_t>>{}).first->second;
	neiIdx.resize(particleParam.size + 1u);

	ComputeNeighborhoodSize_kernel << < mcuda::util::DivUp(particleParam.size, nSearchBlockSize), nSearchBlockSize >> > (
		particleParam.pPosition,
		particleParam.pRadius,
		particleParam.size,
		refParticleParam.pPosition,
		refParticleParam.pRadius,
		pRefHash->GetParam().pStartIdx,
		pRefHash->GetParam().pEndIdx,
		pRefHash->GetParam().pID,
		pRefHash->GetParam().hashSize,
		pRefHash->GetParam().ceilSize,
		thrust::raw_pointer_cast(neiIdx.data()));
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::inclusive_scan(neiIdx.begin(), neiIdx.end(), neiIdx.begin());
	nei.resize(neiIdx.back());

	BuildNeighborhood_kernel << < mcuda::util::DivUp(particleParam.size, nSearchBlockSize), nSearchBlockSize >> > (
		particleParam.pPosition,
		particleParam.pRadius,
		particleParam.size,
		refParticleParam.pPosition,
		refParticleParam.pRadius,
		pRefHash->GetParam().pStartIdx,
		pRefHash->GetParam().pEndIdx,
		pRefHash->GetParam().pID,
		pRefHash->GetParam().hashSize,
		pRefHash->GetParam().ceilSize,
		thrust::raw_pointer_cast(nei.data()),
		thrust::raw_pointer_cast(neiIdx.data()));
	CUDA_CHECK(cudaPeekAtLastError());
}