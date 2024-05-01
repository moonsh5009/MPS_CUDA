#include "stdafx.h"
#include "MPSSpatialHash.cuh"
#include <thrust/sort.h>

#include "MPSSPHParam.h"
#include "MPSBoundaryParticleParam.h"

mps::SpatialHash::SpatialHash() : mps::VirtualTree<SpatialHashParam>{}
{
}

void mps::SpatialHash::SetObjectSize(const size_t size)
{
	m_key.resize(size);
	m_ID.resize(size);

	GetParam().SetObjectSize(size);
	GetParam().SetKeyArray(thrust::raw_pointer_cast(m_key.data()));
	GetParam().SetIDArray(thrust::raw_pointer_cast(m_ID.data()));
}

void mps::SpatialHash::SetCeilSize(const REAL size)
{
	GetParam().SetCeilSize(size);
}

void mps::SpatialHash::SetHashSize(const glm::uvec3& size)
{
	const auto hashCeilSize = size.x * size.y * size.z;
	m_startIdx.resize(hashCeilSize);
	m_endIdx.resize(hashCeilSize);

	GetParam().SetHashSize(size);
	GetParam().SetStartIdxArray(thrust::raw_pointer_cast(m_startIdx.data()));
	GetParam().SetEndIdxArray(thrust::raw_pointer_cast(m_endIdx.data()));
}

std::optional<std::tuple<const uint32_t*, const uint32_t*>> mps::SpatialHash::GetNeighborhood(const ObjectParam& objParam) const
{
	const auto iter = m_mapNei.find(objParam.GetPhaseID());
	if (iter == m_mapNei.end()) return {};
	return std::tuple{ thrust::raw_pointer_cast(iter->second.data.data()), thrust::raw_pointer_cast(iter->second.idx.data()) };
}

void mps::SpatialHash::UpdateHash(const mps::ObjectParam& objParam)
{
	constexpr auto nBlockSize = 1024u;

	const auto nSize = objParam.GetSize();
	if (nSize == 0) return;

	thrust::fill(m_startIdx.begin(), m_startIdx.end(), 0xffffffff);
	thrust::fill(m_endIdx.begin(), m_endIdx.end(), 0xffffffff);

	InitHash_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(GetParam(), objParam);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort_by_key(m_key.begin(), m_key.end(), m_ID.begin());

	ReorderHash_kernel << <mcuda::util::DivUp(nSize, nBlockSize), nBlockSize, (nBlockSize + 1) * sizeof(uint32_t) >> >
		(GetParam());
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::SpatialHash::ZSort(mps::ObjectParam& objParam)
{
	constexpr auto nBlockSize = 1024u;

	const auto nSize = objParam.GetSize();
	if (nSize == 0) return;

	thrust::fill(m_startIdx.begin(), m_startIdx.end(), 0xffffffff);
	thrust::fill(m_endIdx.begin(), m_endIdx.end(), 0xffffffff);

	InitHashZIndex_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(GetParam(), objParam);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort_by_key(m_key.begin(), m_key.end(), thrust::make_zip_iterator(
		thrust::device_pointer_cast(objParam.GetPosArray()),
		thrust::device_pointer_cast(objParam.GetMassArray()),
		thrust::device_pointer_cast(objParam.GetVelocityArray()),
		thrust::device_pointer_cast(objParam.GetColorArray())));
}

void mps::SpatialHash::ZSort(mps::SPHParam& sphParam)
{
	constexpr auto nBlockSize = 1024u;

	const auto nSize = sphParam.GetSize();
	if (nSize == 0) return;

	thrust::fill(m_startIdx.begin(), m_startIdx.end(), 0xffffffff);
	thrust::fill(m_endIdx.begin(), m_endIdx.end(), 0xffffffff);

	InitHashZIndex_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(GetParam(), sphParam);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort_by_key(m_key.begin(), m_key.end(), thrust::make_zip_iterator(
		thrust::device_pointer_cast(sphParam.GetPosArray()),
		thrust::device_pointer_cast(sphParam.GetMassArray()),
		thrust::device_pointer_cast(sphParam.GetVelocityArray()),
		thrust::device_pointer_cast(sphParam.GetColorArray()),
		thrust::device_pointer_cast(sphParam.GetRadiusArray())));
}

void mps::SpatialHash::ZSort(mps::BoundaryParticleParam& boundaryParticleParam)
{
	constexpr auto nBlockSize = 1024u;

	const auto nSize = boundaryParticleParam.GetSize();
	if (nSize == 0) return;

	InitHashZIndex_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(GetParam(), boundaryParticleParam);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort_by_key(m_key.begin(), m_key.end(), thrust::make_zip_iterator(
		thrust::device_pointer_cast(boundaryParticleParam.GetPosArray()),
		thrust::device_pointer_cast(boundaryParticleParam.GetMassArray()),
		thrust::device_pointer_cast(boundaryParticleParam.GetVelocityArray()),
		thrust::device_pointer_cast(boundaryParticleParam.GetColorArray()),
		thrust::device_pointer_cast(boundaryParticleParam.GetRadiusArray()),
		thrust::device_pointer_cast(boundaryParticleParam.GetFaceIDArray()),
		thrust::device_pointer_cast(boundaryParticleParam.GetBCCArray())));
}

void mps::SpatialHash::BuildNeighorhood(const ObjectParam& objParam, REAL radius)
{
	constexpr auto nBlockSize = 512u;

	const auto nSize = objParam.GetSize();
	if (nSize == 0) return;

	const auto emp = m_mapNei.emplace(objParam.GetPhaseID(), NeiBuffer{});
	auto& nei = emp.first->second;
	nei.idx.resize(objParam.GetSize() + 1u);

	ComputeNeighborhoodSize_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >(
		thrust::raw_pointer_cast(nei.idx.data()), radius, objParam, GetParam());
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::inclusive_scan(nei.idx.begin(), nei.idx.end(), nei.idx.begin());
	nei.data.resize(nei.idx.back());

	BuildNeighborhood_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >(
		thrust::raw_pointer_cast(nei.data.data()), thrust::raw_pointer_cast(nei.idx.data()), radius, objParam, GetParam());
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::SpatialHash::BuildNeighorhood(const ObjectParam& objParam, REAL radius, const ObjectParam& refObjParam, const SpatialHash* pRefHash)
{
	constexpr auto nBlockSize = 512u;

	const auto nSize = objParam.GetSize();
	if (nSize == 0 || refObjParam.GetSize() == 0) return;

	const auto emp = m_mapNei.emplace(refObjParam.GetPhaseID(), NeiBuffer{});
	auto& nei = emp.first->second;
	nei.idx.resize(objParam.GetSize() + 1u);

	ComputeNeighborhoodSize_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
		thrust::raw_pointer_cast(nei.idx.data()), radius, objParam, refObjParam, pRefHash->GetParam());
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::inclusive_scan(nei.idx.begin(), nei.idx.end(), nei.idx.begin());
	nei.data.resize(nei.idx.back());

	BuildNeighborhood_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> > (
		thrust::raw_pointer_cast(nei.data.data()), thrust::raw_pointer_cast(nei.idx.data()), radius, objParam, refObjParam, pRefHash->GetParam());
	CUDA_CHECK(cudaPeekAtLastError());
}
