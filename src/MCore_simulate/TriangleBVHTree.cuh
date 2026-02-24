#pragma once

#include "../MCore_util/MCudaUtil.cuh"

#include <cuda/std/detail/libcxx/include/optional>
#include <cuda/std/detail/libcxx/include/tuple>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "TriangleBVHTree.h"

using namespace mcore;

struct BVHElementInfo_CMP
{
	MCUDA_HOST_DEVICE_FUNC bool operator()(const simulate::BVHElementInfo& a, const simulate::BVHElementInfo& b)
	{
		if (a.treeIndex != b.treeIndex) return a.treeIndex < b.treeIndex;
		if (a.zIndex != b.zIndex)		return a.zIndex < b.zIndex;
		return a.index < b.index;
	}
};

MCUDA_DEVICE_FUNC MCUDA_FORCE_INLINE cuda::std::tuple<cuda::std::optional<simulate::BVHNode>, cuda::std::optional<simulate::BVHNode>> GetBVHChildNodes(
	const simulate::DeviceBVHTreeData& bvh,
	size_t nodeIndex,
	IndexType nodeOffset,
	IndexType nodeCount)
{
	const auto localNodeIndex = static_cast<IndexType>(nodeIndex - nodeOffset);
	const auto localChildLeftIndex = (localNodeIndex << 1u) + 1;
	const auto localChildRightIndex = localChildLeftIndex + 1;
	cuda::std::optional<simulate::BVHNode> leftChild;
	if (localChildLeftIndex < nodeCount)
	{
		leftChild = bvh.nodes[nodeOffset + localChildLeftIndex];
	}
	cuda::std::optional<simulate::BVHNode> rightChild;
	if (localChildRightIndex < nodeCount)
	{
		rightChild = bvh.nodes[nodeOffset + localChildRightIndex];
	}
	return { leftChild, rightChild };
}

MCUDA_DEVICE_FUNC MCUDA_FORCE_INLINE int3 GetGridPos(const Vector3& x, REAL radius)
{
	radius = 1.0 / radius;
	return {
		static_cast<int>(x.x() * radius),
		static_cast<int>(x.y() * radius),
		static_cast<int>(x.z() * radius)
	};
}

#pragma nv_diag_suppress 63
#pragma warning(push)
#pragma warning(disable: 4293)
MCUDA_DEVICE_FUNC constexpr IndexType SplitBy3(IndexType x)
{
	if constexpr (std::is_same_v<IndexType, uint64_t>)
	{
		x = x & 0x1fffff;
		x = (x | x << 32ull) & 0x1f00000000ffff;
		x = (x | x << 16ull) & 0x1f0000ff0000ff;
		x = (x | x << 8ull) & 0x100f00f00f00f00f;
		x = (x | x << 4ull) & 0x10c30c30c30c30c3;
		x = (x | x << 2ull) & 0x1249249249249249;
	}
	else
	{
		if (x == 1024u) --x;
		x = (x | x << 16u) & 0b00000011000000000000000011111111;
		x = (x | x << 8u) & 0b00000011000000001111000000001111;
		x = (x | x << 4u) & 0b00000011000011000011000011000011;
		x = (x | x << 2u) & 0b00001001001001001001001001001001;
	}
	return x;
}
#pragma warning(pop)
#pragma nv_diag_default 63

MCUDA_DEVICE_FUNC constexpr IndexType GetZindex(int3 p, uint3 size)
{
	const auto x = (static_cast<IndexType>(p.x + (size.x >> 1u))) & (size.x - 1u);
	const auto y = (static_cast<IndexType>(p.y + (size.y >> 1u))) & (size.y - 1u);
	const auto z = (static_cast<IndexType>(p.z + (size.z >> 1u))) & (size.z - 1u);
	return SplitBy3(x) | SplitBy3(y) << 1u | SplitBy3(z) << 2u;
}

template<simulate::BVH_ELEMENT_TYPE ELEMENT_TYPE>
MCUDA_DEVICE_FUNC simulate::BVHElementInfo GetElementInfo(
	size_t tid,
	const mcuda::DeviceMultiArrayInfo& indexInfo,
	const IndexType* indices,
	const mcuda::DeviceMultiArrayInfo& nodeInfo,
	const Vector3* nodes)
{
	IndexType treeIndex = 0;
	IndexType index = 0;
	IndexType zIndex = 0;

	if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::TRIANGLE)
	{
		const auto baseIdx = tid * 3;
		treeIndex = indexInfo.rangeIndexOfValues[baseIdx];

		const auto indexOffset = indexInfo.rangeOffsets[treeIndex];
		const auto nodeOffset = nodeInfo.rangeOffsets[treeIndex];

		const auto idx0 = nodeOffset + indices[baseIdx];
		const auto idx1 = nodeOffset + indices[baseIdx + 1];
		const auto idx2 = nodeOffset + indices[baseIdx + 2];

		const auto p0 = nodes[idx0];
		const auto p1 = nodes[idx1];
		const auto p2 = nodes[idx2];

		const auto centor = (p0 + p1 + p2) * (1.0 / 3.0);
		const auto gridPos = GetGridPos(centor, 0.5);

		index = tid - indexOffset / 3;
		zIndex = GetZindex(gridPos, make_uint3(512u, 512u, 512u));
	}
	else if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::LINE)
	{
		const auto baseIdx = tid * 2;
		treeIndex = indexInfo.rangeIndexOfValues[baseIdx];

		const auto indexOffset = indexInfo.rangeOffsets[treeIndex];
		const auto nodeOffset = nodeInfo.rangeOffsets[treeIndex];

		const auto idx0 = nodeOffset + indices[baseIdx];
		const auto idx1 = nodeOffset + indices[baseIdx + 1];

		const auto p0 = nodes[idx0];
		const auto p1 = nodes[idx1];

		const auto centor = (p0 + p1) * 0.5;
		const auto gridPos = GetGridPos(centor, 0.5);

		index = tid - indexOffset / 2;
		zIndex = GetZindex(gridPos, make_uint3(512u, 512u, 512u));
	}

	simulate::BVHElementInfo info;
	info.treeIndex = treeIndex;
	info.index = index;
	info.zIndex = zIndex;
	return info;
 }

template<simulate::BVH_ELEMENT_TYPE ELEMENT_TYPE>
MCUDA_DEVICE_FUNC simulate::BVHNode BuildBVHNode(
	IndexType treeIndex,
	size_t nodeIndex,
	size_t nodeCount,
	size_t elementOffset,
	size_t elementCount,
	const mcuda::DeviceMultiArrayInfo& indexInfo,
	const simulate::BVHElementInfo* elementInfos,
	const simulate::DeviceBVHTreeData& bvh)
{
	simulate::BVHNode node;
	node.depth = Log2(nodeIndex + 1);
	node.elementIndex = simulate::BVH_INVALID_INDEX;

	const auto leafNodeIndex = static_cast<IndexType>(nodeCount - elementCount);
	if (nodeIndex >= leafNodeIndex)
	{
		auto localElementIndex = nodeIndex - leafNodeIndex + bvh.treeInfos[treeIndex].pivot;
		if (localElementIndex >= elementCount)
			localElementIndex -= elementCount;

		node.elementIndex = elementInfos[localElementIndex + elementOffset].index;
	}

	return node;
}

template<simulate::BVH_ELEMENT_TYPE ELEMENT_TYPE>
MCUDA_DEVICE_FUNC void RefitLeafNode(
	IndexType treeIndex,
	const mcuda::DeviceMultiArrayInfo& indexInfo,
	const IndexType* indices,
	const mcuda::DeviceMultiArrayInfo& nodeInfo,
	const Vector3* nodes,
	const Vector3& aabbOffset,
	simulate::BVHNode& node)
{
	if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::TRIANGLE)
	{
		const auto indexOffset = indexInfo.rangeOffsets[treeIndex];
		const auto nodeOffset = nodeInfo.rangeOffsets[treeIndex];

		const auto baseIdx = indexOffset + node.elementIndex * 3;

		const auto idx0 = nodeOffset + indices[baseIdx];
		const auto idx1 = nodeOffset + indices[baseIdx + 1];
		const auto idx2 = nodeOffset + indices[baseIdx + 2];

		const auto p0 = nodes[idx0];
		const auto p1 = nodes[idx1];
		const auto p2 = nodes[idx2];

		node.min = p0.cwiseMin(p1).cwiseMin(p2) - aabbOffset;
		node.max = p0.cwiseMax(p1).cwiseMax(p2) + aabbOffset;
	}
	else if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::LINE)
	{
		const auto indexOffset = indexInfo.rangeOffsets[treeIndex];
		const auto nodeOffset = nodeInfo.rangeOffsets[treeIndex];

		const auto baseIdx = indexOffset + node.elementIndex * 2;

		const auto idx0 = nodeOffset + indices[baseIdx];
		const auto idx1 = nodeOffset + indices[baseIdx + 1];

		const auto p0 = nodes[idx0];
		const auto p1 = nodes[idx1];

		node.min = p0.cwiseMin(p1) - aabbOffset;
		node.max = p0.cwiseMax(p1) + aabbOffset;
	}
}

MCUDA_DEVICE_FUNC MCUDA_FORCE_INLINE void GenerateAABBRenderData(
	size_t nodeIndex,
	IndexType nodeOffset,
	const Vector3& aabbMin,
	const Vector3& aabbMax,
	IndexType* lineIndices,
	Vector3* linePositions)
{
	auto baseVertexIdx = static_cast<IndexType>(nodeIndex * 8);
	linePositions[baseVertexIdx + 0] = aabbMin;
	linePositions[baseVertexIdx + 1] = { aabbMax.x(), aabbMin.y(), aabbMin.z() };
	linePositions[baseVertexIdx + 2] = { aabbMax.x(), aabbMax.y(), aabbMin.z() };
	linePositions[baseVertexIdx + 3] = { aabbMin.x(), aabbMax.y(), aabbMin.z() };
	linePositions[baseVertexIdx + 4] = { aabbMin.x(), aabbMin.y(), aabbMax.z() };
	linePositions[baseVertexIdx + 5] = { aabbMax.x(), aabbMin.y(), aabbMax.z() };
	linePositions[baseVertexIdx + 6] = aabbMax;
	linePositions[baseVertexIdx + 7] = { aabbMin.x(), aabbMax.y(), aabbMax.z() };

	baseVertexIdx -= nodeOffset * 8;

	const auto baseIndexIdx = nodeIndex * 24;
	lineIndices[baseIndexIdx + 0] = baseVertexIdx + 0;
	lineIndices[baseIndexIdx + 1] = baseVertexIdx + 1;
	lineIndices[baseIndexIdx + 2] = baseVertexIdx + 1;
	lineIndices[baseIndexIdx + 3] = baseVertexIdx + 2;
	lineIndices[baseIndexIdx + 4] = baseVertexIdx + 2;
	lineIndices[baseIndexIdx + 5] = baseVertexIdx + 3;
	lineIndices[baseIndexIdx + 6] = baseVertexIdx + 3;
	lineIndices[baseIndexIdx + 7] = baseVertexIdx + 0;
	lineIndices[baseIndexIdx + 8] = baseVertexIdx + 4;
	lineIndices[baseIndexIdx + 9] = baseVertexIdx + 5;
	lineIndices[baseIndexIdx + 10] = baseVertexIdx + 5;
	lineIndices[baseIndexIdx + 11] = baseVertexIdx + 6;
	lineIndices[baseIndexIdx + 12] = baseVertexIdx + 6;
	lineIndices[baseIndexIdx + 13] = baseVertexIdx + 7;
	lineIndices[baseIndexIdx + 14] = baseVertexIdx + 7;
	lineIndices[baseIndexIdx + 15] = baseVertexIdx + 4;
	lineIndices[baseIndexIdx + 16] = baseVertexIdx + 0;
	lineIndices[baseIndexIdx + 17] = baseVertexIdx + 4;
	lineIndices[baseIndexIdx + 18] = baseVertexIdx + 1;
	lineIndices[baseIndexIdx + 19] = baseVertexIdx + 5;
	lineIndices[baseIndexIdx + 20] = baseVertexIdx + 2;
	lineIndices[baseIndexIdx + 21] = baseVertexIdx + 6;
	lineIndices[baseIndexIdx + 22] = baseVertexIdx + 3;
	lineIndices[baseIndexIdx + 23] = baseVertexIdx + 7;
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD, simulate::BVH_ELEMENT_TYPE ELEMENT_TYPE>
__global__ void kernel_InitElementInfo(
	IndexType treeIndex,
	const mcuda::DeviceMultiArrayInfo indexInfo,
	const IndexType* indices,
	const mcuda::DeviceMultiArrayInfo nodeInfo,
	const Vector3* nodes,
	simulate::BVHElementInfo* elementInfos)
{
	const auto indexOffset = indexInfo.rangeOffsets[treeIndex];
	const auto indexCount = indexInfo.rangeOffsets[treeIndex + 1] - indexOffset;

	if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::TRIANGLE)
	{
		mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(indexCount / 3, [&](size_t tid)
		{
			elementInfos[tid] = GetElementInfo<ELEMENT_TYPE>(tid + indexOffset / 3, indexInfo, indices, nodeInfo, nodes);
		});
	}
	else if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::LINE)
	{
		mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(indexCount / 2, [&](size_t tid)
		{
			elementInfos[tid] = GetElementInfo<ELEMENT_TYPE>(tid + indexOffset / 2, indexInfo, indices, nodeInfo, nodes);
		});
	}
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD, simulate::BVH_ELEMENT_TYPE ELEMENT_TYPE>
__global__ void kernel_InitElementInfoAll(
	const mcuda::DeviceMultiArrayInfo indexInfo,
	const IndexType* indices,
	const mcuda::DeviceMultiArrayInfo nodeInfo,
	const Vector3* nodes,
	simulate::BVHElementInfo* elementInfos)
{
	if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::TRIANGLE)
	{
		mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(indexInfo.valueCount / 3, [&](size_t tid)
		{
			elementInfos[tid] = GetElementInfo<ELEMENT_TYPE>(tid, indexInfo, indices, nodeInfo, nodes);
		});
	}
	else if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::LINE)
	{
		mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(indexInfo.valueCount / 2, [&](size_t tid)
		{
			elementInfos[tid] = GetElementInfo<ELEMENT_TYPE>(tid, indexInfo, indices, nodeInfo, nodes);
		});
	}
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD, simulate::BVH_ELEMENT_TYPE ELEMENT_TYPE>
__global__ void kernel_BuildBVHTree(
	IndexType treeIndex,
	const mcuda::DeviceMultiArrayInfo indexInfo,
	const simulate::BVHElementInfo* elementInfos,
	simulate::DeviceBVHTreeData bvh)
{
	const auto nodeOffset = bvh.nodeInfo.rangeOffsets[treeIndex];
	const auto nodeCount = bvh.nodeInfo.rangeOffsets[treeIndex + 1] - nodeOffset;
	const auto elementCount = indexInfo.rangeOffsets[treeIndex + 1] - indexInfo.rangeOffsets[treeIndex];

	if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::TRIANGLE)
	{
		mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
		{
			bvh.nodes[nodeOffset + tid] = BuildBVHNode<ELEMENT_TYPE>(
				treeIndex,
				tid,
				nodeCount,
				0,
				elementCount / 3,
				indexInfo,
				elementInfos,
				bvh);
		});
	}
	else if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::LINE)
	{
		mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
		{
			bvh.nodes[nodeOffset + tid] = BuildBVHNode<ELEMENT_TYPE>(
				treeIndex,
				tid,
				nodeCount,
				0,
				elementCount / 2,
				indexInfo,
				elementInfos,
				bvh);
		});
	}
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD, simulate::BVH_ELEMENT_TYPE ELEMENT_TYPE>
__global__ void kernel_BuildBVHTreeAll(
	const mcuda::DeviceMultiArrayInfo indexInfo,
	const simulate::BVHElementInfo* elementInfos,
	simulate::DeviceBVHTreeData bvh)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(bvh.nodeInfo.valueCount, [&](size_t tid)
	{
		const auto treeIndex = bvh.nodeInfo.rangeIndexOfValues[tid];
		const auto nodeOffset = bvh.nodeInfo.rangeOffsets[treeIndex];
		const auto nodeCount = bvh.nodeInfo.rangeOffsets[treeIndex + 1] - nodeOffset;
		const auto elementOffset = indexInfo.rangeOffsets[treeIndex];
		const auto elementCount = indexInfo.rangeOffsets[treeIndex + 1] - elementOffset;

		if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::TRIANGLE)
		{
			bvh.nodes[tid] = BuildBVHNode<ELEMENT_TYPE>(
				treeIndex,
				tid - nodeOffset,
				nodeCount,
				elementOffset / 3,
				elementCount / 3,
				indexInfo,
				elementInfos,
				bvh);
		}
		else if constexpr (ELEMENT_TYPE == simulate::BVH_ELEMENT_TYPE::LINE)
		{
			bvh.nodes[tid] = BuildBVHNode<ELEMENT_TYPE>(
				treeIndex,
				tid - nodeOffset,
				nodeCount,
				elementOffset / 2,
				elementCount / 2,
				indexInfo,
				elementInfos,
				bvh);
		}
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD, simulate::BVH_ELEMENT_TYPE ELEMENT_TYPE>
__global__ void kernel_RefitBVHTreeLeaf(
	const mcuda::DeviceMultiArrayInfo indexInfo,
	const IndexType* indices,
	const mcuda::DeviceMultiArrayInfo nodeInfo,
	const Vector3* nodes,
	simulate::DeviceBVHTreeData bvh)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(bvh.nodeInfo.valueCount, [&](size_t tid)
	{
		const auto treeIndex = bvh.nodeInfo.rangeIndexOfValues[tid];
		const auto aabbOffset = bvh.treeInfos[treeIndex].aabbOffset;

		auto node = bvh.nodes[tid];
		if (node.elementIndex != simulate::BVH_INVALID_INDEX)
		{
			RefitLeafNode<ELEMENT_TYPE>(
				treeIndex,
				indexInfo,
				indices,
				nodeInfo,
				nodes,
				aabbOffset,
				node);
			bvh.nodes[tid] = node;
		}
	});
}

template<simulate::BVH_ELEMENT_TYPE ELEMENT_TYPE>
__global__ void kernel_RefitBVHTreeCG(
	const mcuda::DeviceMultiArrayInfo indexInfo,
	const IndexType* indices,
	const mcuda::DeviceMultiArrayInfo nodeInfo,
	const Vector3* nodes,
	simulate::DeviceBVHTreeData bvh,
	uint32_t maxDepthAcrossAllTrees,
	size_t elementsPerThread)
{
	cg::grid_group grid = cg::this_grid();

	mcuda::kernel_LoopCG(elementsPerThread, bvh.nodeInfo.valueCount, [&](size_t tid)
	{
		const auto treeIndex = bvh.nodeInfo.rangeIndexOfValues[tid];
		const auto aabbOffset = bvh.treeInfos[treeIndex].aabbOffset;

		auto node = bvh.nodes[tid];
		if (node.elementIndex != simulate::BVH_INVALID_INDEX)
		{
			RefitLeafNode<ELEMENT_TYPE>(
				treeIndex,
				indexInfo,
				indices,
				nodeInfo,
				nodes,
				aabbOffset,
				node);
			bvh.nodes[tid] = node;
		}
	});

	for (int depth = maxDepthAcrossAllTrees; depth >= 0; --depth)
	{
		grid.sync();

		mcuda::kernel_LoopCG(elementsPerThread, bvh.nodeInfo.valueCount, [&](size_t tid)
		{
			const auto treeIndex = bvh.nodeInfo.rangeIndexOfValues[tid];
			const auto nodeOffset = bvh.nodeInfo.rangeOffsets[treeIndex];
			const auto nodeCount = bvh.nodeInfo.rangeOffsets[treeIndex + 1] - nodeOffset;
			const auto maxDepth = bvh.treeInfos[treeIndex].maxDepth;

			if (static_cast<uint32_t>(depth) >= maxDepth) return;

			auto node = bvh.nodes[tid];
			if (node.depth != depth || node.elementIndex != simulate::BVH_INVALID_INDEX)
				return;

			const auto [leftChildNode, rightChildNode] = GetBVHChildNodes(bvh, tid, nodeOffset, nodeCount);
			if (leftChildNode)
			{
				node.min = leftChildNode->min;
				node.max = leftChildNode->max;
			}
			if (rightChildNode)
			{
				node.min = node.min.cwiseMin(rightChildNode->min);
				node.max = node.max.cwiseMax(rightChildNode->max);
			}

			bvh.nodes[tid] = node;
		});
	}
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_GetTotalAABB(
	const simulate::DeviceBVHTreeData bvh,
	AABB<REAL>* totalAABB)
{
	__shared__ REAL sharedMinX[BLOCK_SIZE];
	__shared__ REAL sharedMinY[BLOCK_SIZE];
	__shared__ REAL sharedMinZ[BLOCK_SIZE];
	__shared__ REAL sharedMaxX[BLOCK_SIZE];
	__shared__ REAL sharedMaxY[BLOCK_SIZE];
	__shared__ REAL sharedMaxZ[BLOCK_SIZE];

	const auto setSharedMin = [&](const Vector3& v)
	{
		if (v.x() < sharedMinX[threadIdx.x])
			sharedMinX[threadIdx.x] = v.x();
		if (v.y() < sharedMinY[threadIdx.x])
			sharedMinY[threadIdx.x] = v.y();
		if (v.z() < sharedMinZ[threadIdx.x])
			sharedMinZ[threadIdx.x] = v.z();
	};
	const auto setSharedMax = [&](const Vector3& v)
	{
		if (v.x() > sharedMaxX[threadIdx.x])
			sharedMaxX[threadIdx.x] = v.x();
		if (v.y() > sharedMaxY[threadIdx.x])
			sharedMaxY[threadIdx.x] = v.y();
		if (v.z() > sharedMaxZ[threadIdx.x])
			sharedMaxZ[threadIdx.x] = v.z();
	};
	sharedMinX[threadIdx.x] = cuda::std::numeric_limits<REAL>::max();
	sharedMinY[threadIdx.x] = cuda::std::numeric_limits<REAL>::max();
	sharedMinZ[threadIdx.x] = cuda::std::numeric_limits<REAL>::max();
	sharedMaxX[threadIdx.x] = -cuda::std::numeric_limits<REAL>::max();
	sharedMaxY[threadIdx.x] = -cuda::std::numeric_limits<REAL>::max();
	sharedMaxZ[threadIdx.x] = -cuda::std::numeric_limits<REAL>::max();

	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(bvh.nodeInfo.rangeCount, [&](size_t tid)
	{
		const auto treeInfo = bvh.treeInfos[tid];
		const auto nodeOffset = bvh.nodeInfo.rangeOffsets[tid];
		const auto rootNode = bvh.nodes[nodeOffset];
		setSharedMin(rootNode.min);
		setSharedMax(rootNode.max);
	});

	for (size_t offset= BLOCK_SIZE / 2; offset > 32; offset >>= 1)
	{
		__syncthreads();
		if (threadIdx.x < offset)
		{
			setSharedMin({
				sharedMinX[threadIdx.x + offset],
				sharedMinY[threadIdx.x + offset],
				sharedMinZ[threadIdx.x + offset] });
			setSharedMax({
				sharedMaxX[threadIdx.x + offset],
				sharedMaxY[threadIdx.x + offset],
				sharedMaxZ[threadIdx.x + offset] });
		}
	}
	__syncthreads();
	mcuda::WarpMin(sharedMinX, threadIdx.x);
	mcuda::WarpMin(sharedMinY, threadIdx.x);
	mcuda::WarpMin(sharedMinZ, threadIdx.x);
	mcuda::WarpMax(sharedMaxX, threadIdx.x);
	mcuda::WarpMax(sharedMaxY, threadIdx.x);
	mcuda::WarpMax(sharedMaxZ, threadIdx.x);
	if (threadIdx.x == 0)
	{
		mcuda::AtomicMin(&totalAABB->GetMin().x, sharedMinX[0]);
		mcuda::AtomicMin(&totalAABB->GetMin().y, sharedMinY[0]);
		mcuda::AtomicMin(&totalAABB->GetMin().z, sharedMinZ[0]);
		mcuda::AtomicMax(&totalAABB->GetMax().x, sharedMaxX[0]);
		mcuda::AtomicMax(&totalAABB->GetMax().y, sharedMaxY[0]);
		mcuda::AtomicMax(&totalAABB->GetMax().z, sharedMaxZ[0]);
	}
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_UpdateBVHRenderBuffers(
	const simulate::DeviceBVHTreeData bvh,
	const AABB<REAL>* totalAABB,
	IndexType* lineIndices,
	Vector3* linePositions)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(bvh.nodeInfo.valueCount, [&](size_t tid)
	{
		const auto treeIndex = bvh.nodeInfo.rangeIndexOfValues[tid];
		const auto nodeOffset = bvh.nodeInfo.rangeOffsets[treeIndex];
		const auto node = bvh.nodes[tid];

		/*if (tid != 1)
		{
			const auto baseIndexIdx = tid * 24;
			lineIndices[baseIndexIdx + 0] = 0;
			lineIndices[baseIndexIdx + 1] = 0;
			lineIndices[baseIndexIdx + 2] = 0;
			lineIndices[baseIndexIdx + 3] = 0;
			lineIndices[baseIndexIdx + 4] = 0;
			lineIndices[baseIndexIdx + 5] = 0;
			lineIndices[baseIndexIdx + 6] = 0;
			lineIndices[baseIndexIdx + 7] = 0;
			lineIndices[baseIndexIdx + 8] = 0;
			lineIndices[baseIndexIdx + 9] = 0;
			lineIndices[baseIndexIdx + 10] = 0;
			lineIndices[baseIndexIdx + 11] = 0;
			lineIndices[baseIndexIdx + 12] = 0;
			lineIndices[baseIndexIdx + 13] = 0;
			lineIndices[baseIndexIdx + 14] = 0;
			lineIndices[baseIndexIdx + 15] = 0;
			lineIndices[baseIndexIdx + 16] = 0;
			lineIndices[baseIndexIdx + 17] = 0;
			lineIndices[baseIndexIdx + 18] = 0;
			lineIndices[baseIndexIdx + 19] = 0;
			lineIndices[baseIndexIdx + 20] = 0;
			lineIndices[baseIndexIdx + 21] = 0;
			lineIndices[baseIndexIdx + 22] = 0;
			lineIndices[baseIndexIdx + 23] = 0;
			return;
		}*/

		GenerateAABBRenderData(
			tid,
			nodeOffset,
			node.min,
			node.max,
			lineIndices,
			linePositions);
	});

	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		auto aabbMin = mcuda::Convert(totalAABB->GetMin());
		auto aabbMax = mcuda::Convert(totalAABB->GetMax());

		GenerateAABBRenderData(
			bvh.nodeInfo.valueCount,
			bvh.nodeInfo.valueCount,
			aabbMin,
			aabbMax,
			lineIndices,
			linePositions);
	}
}