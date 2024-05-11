#pragma once
#include "MPSSpatialHash.h"
#include "../MCUDA_Lib/MCUDAHelper.cuh"

#include "MPSObjectParam.h"

namespace mps::device::SpatialHash
{
	MCUDA_DEVICE_FUNC glm::ivec3 GetGridPos(const REAL3& x, const glm::uvec3& hashSize, REAL ceilSize)
	{
		const auto invRadius = 1.0 / ceilSize;
		glm::ivec3 p =
		{
			static_cast<int>(x.x * invRadius) + (hashSize.x >> 1),
			static_cast<int>(x.y * invRadius) + (hashSize.y >> 1),
			static_cast<int>(x.z * invRadius) + (hashSize.z >> 1)
		};
		return p;
	}

	MCUDA_DEVICE_FUNC uint32_t GetGridIndex(const glm::ivec3& p, const glm::uvec3& hashSize)
	{
		return __umul24(__umul24(
			static_cast<uint32_t>(p.z) & (hashSize.z - 1), hashSize.y) +
			(static_cast<uint32_t>(p.y) & (hashSize.y - 1)), hashSize.x) +
			(static_cast<uint32_t>(p.x) & (hashSize.x - 1));
	}

	MCUDA_DEVICE_FUNC uint32_t GetGridZIndex(const glm::ivec3& p, const glm::uvec3& hashSize)
	{
		const auto x = static_cast<uint32_t>(p.x + (hashSize.x >> 1u)) & (hashSize.x - 1u);
		const auto y = static_cast<uint32_t>(p.y + (hashSize.y >> 1u)) & (hashSize.y - 1u);
		const auto z = static_cast<uint32_t>(p.z + (hashSize.z >> 1u)) & (hashSize.z - 1u);

		constexpr auto SplitBy3 = [](uint32_t x) -> uint32_t
		{
			if (x == 1024u) x--;
			x = (x | x << 16u) & 0b00000011000000000000000011111111;
			x = (x | x << 8u) & 0b00000011000000001111000000001111;
			x = (x | x << 4u) & 0b00000011000011000011000011000011;
			x = (x | x << 2u) & 0b00001001001001001001001001001001;
			return x;
		};
		return SplitBy3(x) | (SplitBy3(y) << 1u) | (SplitBy3(z) << 2u);
	}

	template<class Fn>
	MCUDA_DEVICE_FUNC void Research(
		const uint32_t* pStartIdx,
		const uint32_t* pEndIdx,
		const uint32_t* pID,
		const glm::uvec3& hashSize,
		REAL ceilSize,
		const REAL3& pos,
		Fn func)
	{
		const auto p = GetGridPos(pos, hashSize, ceilSize);
		glm::ivec3 q;
	#pragma unroll
		for (q.z = p.z - 1; q.z <= p.z + 1; q.z++)
		{
		#pragma unroll
			for (q.y = p.y - 1; q.y <= p.y + 1; q.y++)
			{
			#pragma unroll
				for (q.x = p.x - 1; q.x <= p.x + 1; q.x++)
				{
					const auto idxGrid = GetGridIndex(q, hashSize);

					auto iStart = pStartIdx[idxGrid];
					const auto iEnd = pEndIdx[idxGrid];
				#pragma unroll
					for (; iStart < iEnd; iStart++)
					{
						func(pID[iStart]);
					}
				}
			}
		}
	}
}

__global__ void InitHash_kernel(
	const REAL3* MCUDA_RESTRICT pObjPosition,
	size_t objSize,
	uint32_t* MCUDA_RESTRICT pHashKey,
	uint32_t* MCUDA_RESTRICT pHashID,
	glm::uvec3 hashSize,
	REAL hashCeilSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objSize) return;
	
	const auto gridPos = mps::device::SpatialHash::GetGridPos(pObjPosition[id], hashSize, hashCeilSize);
	const auto gridIdx = mps::device::SpatialHash::GetGridIndex(gridPos, hashSize);
	pHashKey[id] = gridIdx;
	pHashID[id] = id;
}

__global__ void InitHashZIndex_kernel(
	const REAL3* MCUDA_RESTRICT pObjPosition,
	size_t objSize,
	uint32_t* MCUDA_RESTRICT pHashKey,
	glm::uvec3 hashSize,
	REAL hashCeilSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objSize) return;

	const auto gridPos = mps::device::SpatialHash::GetGridPos(pObjPosition[id], hashSize, hashCeilSize);
	const auto gridIdx = mps::device::SpatialHash::GetGridZIndex(gridPos, hashSize);
	pHashKey[id] = gridIdx;
}

__global__ void ReorderHash_kernel(
	size_t objSize,
	const uint32_t* MCUDA_RESTRICT pHashKey,
	uint32_t* MCUDA_RESTRICT pHashStartIdx,
	uint32_t* MCUDA_RESTRICT pHashEndIdx)
{
	extern __shared__ uint32_t s_hash[];
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t hashId;

	if (id < objSize)
	{
		hashId = pHashKey[id];
		s_hash[threadIdx.x + 1] = hashId;
		if (id > 0 && threadIdx.x == 0)
			s_hash[0] = pHashKey[id - 1];
	}
	__syncthreads();

	if (id < objSize)
	{
		const auto prev_hashId = s_hash[threadIdx.x];
		if (id == 0 || prev_hashId != hashId) {
			pHashStartIdx[hashId] = id;
			if (id > 0)
				pHashEndIdx[prev_hashId] = id;
		}
		if (id == objSize - 1)
			pHashEndIdx[hashId] = objSize;
	}
}

__global__ void ComputeNeighborhoodSize_kernel(
	const REAL3* MCUDA_RESTRICT pObjPosition,
	const REAL* MCUDA_RESTRICT pObjRadius,
	size_t objSize,
	const uint32_t* MCUDA_RESTRICT pHashStartIdx,
	const uint32_t* MCUDA_RESTRICT pHashEndIdx,
	const uint32_t* MCUDA_RESTRICT pHashID,
	glm::uvec3 hashSize,
	REAL hashCeilSize,
	uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objSize) return;

	const auto xi = pObjPosition[id];
	const auto ri = pObjRadius[id] * mps::device::SPH::H_RATIO;
	uint32_t num = 0u;

	mps::device::SpatialHash::Research(pHashStartIdx, pHashEndIdx, pHashID, hashSize, hashCeilSize, xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = pObjPosition[jd];
		const auto rj = pObjRadius[jd] * mps::device::SPH::H_RATIO;
		const auto dist = glm::length(xi - xj);
		num += static_cast<uint32_t>(dist < (ri + rj) * 0.5);
	});

	if (id == 0) pNeiIdx[0] = 0u;
	pNeiIdx[id + 1u] = num;
}

__global__ void ComputeNeighborhoodSize_kernel(
	const REAL3* MCUDA_RESTRICT pObjPosition,
	const REAL* MCUDA_RESTRICT pObjRadius,
	size_t objSize,
	const REAL3* MCUDA_RESTRICT pRefObjPos,
	const REAL* MCUDA_RESTRICT pRefObjRadius,
	const uint32_t* MCUDA_RESTRICT pRefHashStartIdx,
	const uint32_t* MCUDA_RESTRICT pRefHashEndIdx,
	const uint32_t* MCUDA_RESTRICT pRefHashID,
	glm::uvec3 refHashSize,
	REAL refHashCeilSize,
	uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objSize) return;

	const auto xi = pObjPosition[id];
	const auto ri = pObjRadius[id] * mps::device::SPH::H_RATIO;
	uint32_t num = 0u;

	mps::device::SpatialHash::Research(pRefHashStartIdx, pRefHashEndIdx, pRefHashID, refHashSize, refHashCeilSize, xi, [&](uint32_t jd)
	{
		const auto xj = pRefObjPos[jd];
		const auto rj = pRefObjRadius[jd] * mps::device::SPH::H_RATIO;
		const auto dist = glm::length(xi - xj);
		num += static_cast<uint32_t>(dist < (ri + rj) * 0.5);
	});

	if (id == 0) pNeiIdx[0] = 0u;
	pNeiIdx[id + 1u] = num;
}

__global__ void BuildNeighborhood_kernel(
	const REAL3* MCUDA_RESTRICT pObjPosition,
	const REAL* MCUDA_RESTRICT pObjRadius,
	size_t objSize,
	const uint32_t* MCUDA_RESTRICT pHashStartIdx,
	const uint32_t* MCUDA_RESTRICT pHashEndIdx,
	const uint32_t* MCUDA_RESTRICT pHashID,
	glm::uvec3 hashSize,
	REAL hashCeilSize,
	uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objSize) return;

	const auto xi = pObjPosition[id];
	const auto ri = pObjRadius[id] * mps::device::SPH::H_RATIO;
	uint32_t idx = pNeiIdx[id];

	mps::device::SpatialHash::Research(pHashStartIdx, pHashEndIdx, pHashID, hashSize, hashCeilSize, xi, [&](uint32_t jd)
	{
		if (id == jd) return;

		const auto xj = pObjPosition[jd];
		const auto rj = pObjRadius[jd] * mps::device::SPH::H_RATIO;
		const auto dist = glm::length(xi - xj);
		if (dist < (ri + rj) * 0.5)
			pNei[idx++] = jd;
	});
}

__global__ void BuildNeighborhood_kernel(
	const REAL3* MCUDA_RESTRICT pObjPosition,
	const REAL* MCUDA_RESTRICT pObjRadius,
	size_t objSize,
	const REAL3* MCUDA_RESTRICT pRefObjPos,
	const REAL* MCUDA_RESTRICT pRefObjRadius,
	const uint32_t* MCUDA_RESTRICT pRefHashStartIdx,
	const uint32_t* MCUDA_RESTRICT pRefHashEndIdx,
	const uint32_t* MCUDA_RESTRICT pRefHashID,
	glm::uvec3 refHashSize,
	REAL refHashCeilSize,
	uint32_t* MCUDA_RESTRICT pNei,
	const uint32_t* MCUDA_RESTRICT pNeiIdx)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objSize) return;

	const auto xi = pObjPosition[id];
	const auto ri = pObjRadius[id] * mps::device::SPH::H_RATIO;
	uint32_t idx = pNeiIdx[id];

	mps::device::SpatialHash::Research(pRefHashStartIdx, pRefHashEndIdx, pRefHashID, refHashSize, refHashCeilSize, xi, [&](uint32_t jd)
	{
		const auto xj = pRefObjPos[jd];
		const auto rj = pRefObjRadius[jd] * mps::device::SPH::H_RATIO;
		const auto dist = glm::length(xi - xj);
		if (dist < (ri + rj) * 0.5)
			pNei[idx++] = jd;
	});
}