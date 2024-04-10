#pragma once
#include "MPSSpatialHash.h"

#include "../MCUDA_Lib/MCUDAHelper.cuh"
#include "MPSObjectParam.h"

MCUDA_DEVICE_FUNC glm::ivec3 mps::SpatialHashParam::GetGridPos(const REAL3& x) const
{
	const auto invRadius = 1.0 / m_ceilSize;
	glm::ivec3 p =
	{
		static_cast<int>(x.x * invRadius) - (m_hashSize.x >> 1),
		static_cast<int>(x.y * invRadius) - (m_hashSize.y >> 1),
		static_cast<int>(x.z * invRadius) - (m_hashSize.z >> 1)
	};
	return p;
}

MCUDA_DEVICE_FUNC uint32_t mps::SpatialHashParam::GetGridIndex(const glm::ivec3& p) const
{
	return __umul24(__umul24
	(static_cast<uint32_t>(p.z) & (m_hashSize.z - 1), m_hashSize.y) +
		(static_cast<uint32_t>(p.y) & (m_hashSize.y - 1)), m_hashSize.x) +
		(static_cast<uint32_t>(p.x) & (m_hashSize.x - 1));
}

MCUDA_DEVICE_FUNC uint32_t mps::SpatialHashParam::GetGridZindex(const glm::ivec3& p) const
{
	const auto x = (static_cast<uint32_t>(p.x + (m_hashSize.x >> 1u))) & (m_hashSize.x - 1u);
	const auto y = (static_cast<uint32_t>(p.y + (m_hashSize.y >> 1u))) & (m_hashSize.y - 1u);
	const auto z = (static_cast<uint32_t>(p.z + (m_hashSize.z >> 1u))) & (m_hashSize.z - 1u);

	const auto SplitBy3 = [](uint32_t x) -> uint32_t
	{
		if (x == 1024u) x--;
		x = (x | x << 16u) & 0b00000011000000000000000011111111;
		x = (x | x << 8u) & 0b00000011000000001111000000001111;
		x = (x | x << 4u) & 0b00000011000011000011000011000011;
		x = (x | x << 2u) & 0b00001001001001001001001001001001;
		return x;
	};
	return SplitBy3(x) | SplitBy3(y) << 1u | SplitBy3(z) << 2u;
}

template<class Fn>
MCUDA_DEVICE_FUNC void mps::SpatialHashParam::Research(const REAL3& pos, Fn func) const
{
	const auto p = GetGridPos(pos);
	glm::ivec3 q;
	for (q.x = p.x - 1; q.x <= p.x + 1; q.x++)
	{
		for (q.y = p.y - 1; q.y <= p.y + 1; q.y++)
		{
			for (q.z = p.z - 1; q.z <= p.z + 1; q.z++)
			{
				const auto idxGrid = GetGridIndex(q);
				auto iStart = m_pStartIdx[idxGrid];
				const auto iEnd = m_pEndIdx[idxGrid];
				for (; iStart < iEnd; iStart++)
				{
					func(m_pID[iStart]);
				}
			}
		}
	}
}

__global__ void InitHash_kernel(const mps::SpatialHashParam hash, const mps::ObjectParam objParam)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objParam.GetSize()) return;
	
	const auto gridPos = hash.GetGridPos(objParam.Position(id));
	const auto gridIdx = hash.GetGridIndex(gridPos);
	hash.Key(id) = gridIdx;
	hash.ID(id) = id;
}

__global__ void ReorderHash_kernel(const mps::SpatialHashParam hash)
{
	extern __shared__ uint32_t s_hash[];
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t hashId;

	if (id < hash.GetObjectSize())
	{
		hashId = hash.Key(id);
		s_hash[threadIdx.x + 1] = hashId;
		if (id > 0 && threadIdx.x == 0)
			s_hash[0] = hash.Key(id - 1);
	}
	__syncthreads();

	if (id < hash.GetObjectSize())
	{
		const auto prev_hashId = s_hash[threadIdx.x];
		if (id == 0 || prev_hashId != hashId) {
			hash.StartIdx(hashId) = id;
			if (id > 0)
				hash.EndIdx(prev_hashId) = id;
		}
		if (id == hash.GetObjectSize() - 1)
			hash.EndIdx(hashId) = hash.GetObjectSize();
	}
}