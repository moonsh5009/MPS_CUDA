#pragma once

#include "../MCore_util/HostSingleArray.h"
#include "../MCore_util/DeviceSingleArray.h"
#include "../MCore_util/AABB.h"
#include "../MCore_util/BufferDef.h"

#include "BVHTreeDef.h"

#include "HeaderPre.h"

namespace mcore::simulate
{
	class __MY_EXT_CLASS__ TriangleBVHTree
	{
	public:
		TriangleBVHTree();
		~TriangleBVHTree() = default;
		TriangleBVHTree(const TriangleBVHTree&) = delete;
		TriangleBVHTree(TriangleBVHTree&&) = default;
		TriangleBVHTree& operator=(const TriangleBVHTree&) = delete;
		TriangleBVHTree& operator=(TriangleBVHTree&&) = default;

		void AddTree(
			const mcuda::VKDeviceMultiArray<IndexType>& indexArray,
			mcuda::DeviceMultiArrayInfo nodeInfo,
			const mcuda::VKDeviceBuffer<mcore::Vector3>& nodes);
		void SetTree(
			IndexType index,
			const mcuda::VKDeviceMultiArray<IndexType>& indexArray,
			mcuda::DeviceMultiArrayInfo nodeInfo,
			const mcuda::VKDeviceBuffer<mcore::Vector3>& nodes);
		void RemoveTree(IndexType index);

		void SetOffset(IndexType index, const Vector3& offset);

		void Build(
			IndexType index,
			const mcuda::VKDeviceMultiArray<IndexType>& indexArray,
			mcuda::DeviceMultiArrayInfo nodeInfo,
			const mcuda::VKDeviceBuffer<mcore::Vector3>& nodes);
		void BuildAll(
			const mcuda::VKDeviceMultiArray<IndexType>& indexArray,
			mcuda::DeviceMultiArrayInfo nodeInfo,
			const mcuda::VKDeviceBuffer<mcore::Vector3>& nodes);

		void Refit(
			const mcuda::VKDeviceMultiArray<IndexType>& indexArray,
			mcuda::DeviceMultiArrayInfo nodeInfo,
			const mcuda::VKDeviceBuffer<mcore::Vector3>& nodes);

		void ResetTotalAABBBuffer();

		void BuildRenderBuffers();
		void UpdateRenderBuffers();

		auto& GetTotalAABBBuffer() { return m_totalAABBBuffer; }
		auto& GetRenderLineIndices() { return lineIndices; }
		auto& GetRenderVertexPositions() { return vertexPositions; }
		auto& GetRenderVertexOffsets() { return vertexOffsets; }
		auto& GetRenderLineDrawIndirects() { return lineDrawIndirects; }

		auto& GetTotalAABBBuffer() const { return m_totalAABBBuffer; }
		auto& GetRenderLineIndices() const { return lineIndices; }
		auto& GetRenderVertexPositions() const { return vertexPositions; }
		auto& GetRenderVertexOffsets() const { return vertexOffsets; }
		auto& GetRenderLineDrawIndirects() const { return lineDrawIndirects; }

		DeviceBVHTreeData GetDeviceData() const
		{
			DeviceBVHTreeData data;
			data.treeInfos = m_bvhTreeInfos.GetBuffer().GetData();
			data.nodes = m_bvhNodes.GetBuffer().GetData();
			data.nodeInfo = m_bvhNodes.GetDeviceInfo();
			return data;
		}

		void SetTotalAABBUpdateFunction(std::function<void(const AABB<REAL>& aabb)>&& func)
		{
			m_totalAABBUpdateFunc = std::move(func);
		}

	private:
		mcuda::VKDeviceBuffer<AABB<REAL>> m_totalAABBBuffer;

		mcuda::HostSingleArray<BVHTreeInfo> m_hostBVHTreeInfos;
		mcuda::DeviceSingleArray<BVHTreeInfo> m_bvhTreeInfos;
		mcuda::DeviceMultiArray<BVHNode> m_bvhNodes;

		mcuda::VKDeviceBuffer<IndexType> lineIndices;
		mcuda::VKDeviceBuffer<mcore::Vector3> vertexPositions;
		mcuda::VKDeviceBuffer<IndexType> vertexOffsets;
		mcuda::VKDeviceBuffer<mvk::DrawIndirectCommand> lineDrawIndirects;

		std::function<void(const AABB<REAL>& aabb)> m_totalAABBUpdateFunc;
	};
}

#include "HeaderPost.h"