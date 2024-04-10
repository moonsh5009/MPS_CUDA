#include "stdafx.h"
#include "MPSSpatialHash.cuh"
#include "thrust/sort.h"

namespace
{
	constexpr auto nBlockSize = 1024u;
}

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

#include "thrust/host_vector.h"
void mps::SpatialHash::UpdateHash(const mps::ObjectParam& objParam)
{
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