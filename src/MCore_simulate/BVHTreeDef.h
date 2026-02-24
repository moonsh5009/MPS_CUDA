#pragma once

#include "../MCore_util/DeviceMultiArray.h"

#include "../MCore_interface/DBDef.h"

namespace mcore::simulate
{
	enum class BVH_ELEMENT_TYPE : uint8_t
	{
		TRIANGLE = 0,
		LINE = 1,
	};

	constexpr IndexType BVH_INVALID_INDEX = static_cast<IndexType>(-1);

	struct BVHElementInfo
	{
		IndexType treeIndex;
		IndexType index;
		IndexType zIndex;
	};

	struct BVHTreeInfo
	{
		uint32_t maxDepth;
		IndexType pivot;
		Vector3 aabbOffset;
	};

	struct BVHNode
	{
		Vector3 min;
		Vector3 max;
		uint32_t depth;
		IndexType elementIndex;
	};

	struct DeviceBVHTreeData
	{
		BVHTreeInfo* treeInfos;
		BVHNode* nodes;
		mcuda::DeviceMultiArrayInfo nodeInfo;
	};

	inline std::tuple<BVHTreeInfo, size_t> GenerateTreeInfo(uint32_t elementCount)
	{
		const auto maxDepth = Log2((elementCount - 1u) << 1u);
		const auto dummy = (1u << maxDepth) - elementCount;
		const auto pivot = ((1u << (maxDepth - 1u)) - dummy) << 1u;
		const auto nodeCount = (1u << (maxDepth + 1u)) - 1u - (dummy << 1u);

		BVHTreeInfo treeInfo;
		treeInfo.maxDepth = maxDepth;
		treeInfo.pivot = pivot;
		return { treeInfo, nodeCount };
	}
}