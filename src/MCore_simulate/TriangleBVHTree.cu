#include "stdafx.h"
#include "TriangleBVHTree.cuh"

#include <thrust/sort.h>

using namespace mcore::simulate;

namespace
{
	constexpr size_t BLOCK_SIZE = 1024;
	constexpr size_t ELEMENTS_PER_THREAD = 8;
	constexpr size_t BLOCK_ELEMENTS = BLOCK_SIZE * ELEMENTS_PER_THREAD;
}

TriangleBVHTree::TriangleBVHTree()
	: m_totalAABBBuffer{ vk::BufferUsageFlagBits::eStorageBuffer }
	, lineIndices{ vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
	, vertexPositions{ vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
	, vertexOffsets{ vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
	, lineDrawIndirects{ vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
{
	m_totalAABBBuffer.SetSize(1);
	ResetTotalAABBBuffer();
}

void TriangleBVHTree::AddTree(
	const mcuda::VKDeviceMultiArray<IndexType>& indexArray,
	mcuda::DeviceMultiArrayInfo nodeInfo,
	const mcuda::VKDeviceBuffer<mcore::Vector3>& nodes)
{
	const auto index = static_cast<IndexType>(m_hostBVHTreeInfos.GetSize());

	const auto elementCount = indexArray.GetRanges().back().size / 3;
	const auto [treeInfo, nodeCount] = GenerateTreeInfo(elementCount);

	m_hostBVHTreeInfos.Insert(treeInfo);
	m_bvhTreeInfos.Insert(treeInfo);
	m_bvhNodes.Insert(std::vector<BVHNode>(nodeCount));

	Build(index, indexArray, nodeInfo, nodes);
}

void TriangleBVHTree::SetTree(
	IndexType index,
	const mcuda::VKDeviceMultiArray<IndexType>& indexArray,
	mcuda::DeviceMultiArrayInfo nodeInfo,
	const mcuda::VKDeviceBuffer<mcore::Vector3>& nodes)
{
	const auto elementCount = indexArray.GetRanges().back().size / 3;
	const auto [treeInfo, nodeCount] = GenerateTreeInfo(elementCount);

	m_hostBVHTreeInfos.Set(index, treeInfo);
	m_bvhTreeInfos.Set(index, treeInfo);
	m_bvhNodes.Set(index, std::vector<BVHNode>(nodeCount));

	Build(index, indexArray, nodeInfo, nodes);
}

void TriangleBVHTree::RemoveTree(IndexType index)
{
	m_hostBVHTreeInfos.Remove(index);
	m_bvhTreeInfos.Remove(index);
	m_bvhNodes.Remove(index);

	BuildRenderBuffers();
}

void TriangleBVHTree::SetOffset(IndexType index, const Vector3& offset)
{
	auto treeInfo = m_hostBVHTreeInfos[index];
	treeInfo.aabbOffset = offset;
	m_hostBVHTreeInfos.Set(index, treeInfo);
	m_bvhTreeInfos.Set(index, treeInfo);
}

void TriangleBVHTree::Build(
	IndexType index,
	const mcuda::VKDeviceMultiArray<IndexType>& indexArray,
	mcuda::DeviceMultiArrayInfo nodeInfo,
	const mcuda::VKDeviceBuffer<mcore::Vector3>& nodes)
{
	const auto elementCount = indexArray.GetRanges()[index].size / 3;
	const auto indexInfo = indexArray.GetDeviceInfo();

	mcuda::DeviceBuffer<BVHElementInfo> elementInfos;
	elementInfos.SetSize(elementCount);

	{
		const auto gridCount = mcore::DivUp<BLOCK_ELEMENTS>(elementCount);
		kernel_InitElementInfo<BLOCK_SIZE, ELEMENTS_PER_THREAD, BVH_ELEMENT_TYPE::TRIANGLE> << <gridCount, BLOCK_SIZE >> > (
			index,
			indexInfo,
			indexArray.GetBuffer().GetData(),
			nodeInfo,
			nodes.GetData(),
			elementInfos.GetData());
		CUDA_CHECK(cudaPeekAtLastError());
	}

	thrust::sort(elementInfos.begin(), elementInfos.end(), BVHElementInfo_CMP());

	{
		const auto gridCount = mcore::DivUp<BLOCK_ELEMENTS>(m_bvhNodes.GetRange(index)->size);
		kernel_BuildBVHTree<BLOCK_SIZE, ELEMENTS_PER_THREAD, BVH_ELEMENT_TYPE::TRIANGLE> << <gridCount, BLOCK_SIZE >> > (
			index,
			indexInfo,
			elementInfos.GetData(),
			GetDeviceData());
		CUDA_CHECK(cudaPeekAtLastError());
	}

	/*{
		const auto totalNodeCount = m_bvhNodes.GetBuffer().GetSize();
		std::vector<BVHNode> hostNodes;
		hostNodes.resize(totalNodeCount);
		m_bvhNodes.GetBuffer().CopyToHost(hostNodes.data(), sizeof(BVHNode) * totalNodeCount);
		for (IndexType i = 0; i < hostNodes.size(); ++i)
		{
			Logger::InfoNoNewLine("Node ", i, ": ");
			Logger::InfoNoNewLine(hostNodes[i].depth, ", ");
			Logger::InfoNoNewLine(hostNodes[i].elementIndex, ", ");
			Logger::InfoNoNewLine("max(", hostNodes[i].max.x(), ", ");
			Logger::InfoNoNewLine(hostNodes[i].max.y(), ", ");
			Logger::InfoNoNewLine(hostNodes[i].max.z(), "), ");
			Logger::InfoNoNewLine("min(", hostNodes[i].min.x(), ", ");
			Logger::InfoNoNewLine(hostNodes[i].min.y(), ", ");
			Logger::Info(hostNodes[i].min.z(), ")");
			Logger::Print();
		}
	}*/

	BuildRenderBuffers();
}

void TriangleBVHTree::BuildAll(
	const mcuda::VKDeviceMultiArray<IndexType>&indexArray,
	mcuda::DeviceMultiArrayInfo nodeInfo,
	const mcuda::VKDeviceBuffer<mcore::Vector3>& nodes)
{
	const auto elementCount = (indexArray.GetRanges().back().offset + indexArray.GetRanges().back().size) / 3;
	const auto indexInfo = indexArray.GetDeviceInfo();

	mcuda::DeviceBuffer<BVHElementInfo> elementInfos;
	elementInfos.SetSize(elementCount);

	{
		const auto gridCount = mcore::DivUp<BLOCK_ELEMENTS>(elementCount);
		kernel_InitElementInfoAll<BLOCK_SIZE, ELEMENTS_PER_THREAD, BVH_ELEMENT_TYPE::TRIANGLE> << <gridCount, BLOCK_SIZE >> > (
			indexInfo,
			indexArray.GetBuffer().GetData(),
			nodeInfo,
			nodes.GetData(),
			elementInfos.GetData());
		CUDA_CHECK(cudaPeekAtLastError());
	}

	thrust::sort(elementInfos.begin(), elementInfos.end(), BVHElementInfo_CMP());

	{
		const auto gridCount = mcore::DivUp<BLOCK_ELEMENTS>(m_bvhNodes.GetBuffer().GetSize());
		kernel_BuildBVHTreeAll<BLOCK_SIZE, ELEMENTS_PER_THREAD, BVH_ELEMENT_TYPE::TRIANGLE> << <gridCount, BLOCK_SIZE >> > (
			indexInfo,
			elementInfos.GetData(),
			GetDeviceData());
		CUDA_CHECK(cudaPeekAtLastError());
	}

	BuildRenderBuffers();
}

void TriangleBVHTree::Refit(
	const mcuda::VKDeviceMultiArray<IndexType>& indexArray,
	mcuda::DeviceMultiArrayInfo nodeInfo,
	const mcuda::VKDeviceBuffer<mcore::Vector3>& nodes)
{
	const auto totalNodeCount = m_bvhNodes.GetBuffer().GetSize();
	if (!totalNodeCount) return;

	const auto indexInfo = indexArray.GetDeviceInfo();
	const auto ptrIndices = indexArray.GetBuffer().GetData();
	const auto ptrNodes = nodes.GetData();
	const auto bvhTreeInfo = GetDeviceData();

	uint32_t maxDepthAcrossAllTrees = 0;
	for (IndexType i = 0; i < m_hostBVHTreeInfos.GetSize(); ++i)
	{
		maxDepthAcrossAllTrees = std::max(maxDepthAcrossAllTrees, m_hostBVHTreeInfos[i].maxDepth);
	}

	const auto maxBlocks = mcuda::GetMaxCooperativeBlocks(0, (void*)kernel_RefitBVHTreeCG<BVH_ELEMENT_TYPE::TRIANGLE>, BLOCK_SIZE);
	const auto dynamicsElementsPerBlock = (totalNodeCount + BLOCK_SIZE * maxBlocks - 1) / (BLOCK_SIZE * maxBlocks);
	const auto gridCount = (totalNodeCount + BLOCK_SIZE * dynamicsElementsPerBlock - 1) / (BLOCK_SIZE * dynamicsElementsPerBlock);

	void* kernelArgs[] = {
		(void*)&indexInfo,
		(void*)&ptrIndices,
		(void*)&nodeInfo,
		(void*)&ptrNodes,
		(void*)&bvhTreeInfo,
		(void*)&maxDepthAcrossAllTrees,
		(void*)&dynamicsElementsPerBlock,
	};

	CUDA_CHECK(cudaLaunchCooperativeKernel(
		reinterpret_cast<void*>(kernel_RefitBVHTreeCG<BVH_ELEMENT_TYPE::TRIANGLE>),
		static_cast<uint32_t>(gridCount),
		static_cast<uint32_t>(BLOCK_SIZE),
		kernelArgs));
	CUDA_CHECK(cudaPeekAtLastError());

	{
		const auto gridCount = mcore::DivUp<BLOCK_ELEMENTS>(m_bvhTreeInfos.GetSize());
		ResetTotalAABBBuffer();

		kernel_GetTotalAABB<BLOCK_SIZE, ELEMENTS_PER_THREAD> << <gridCount, BLOCK_SIZE >> > (
			bvhTreeInfo,
			m_totalAABBBuffer.GetData());
		CUDA_CHECK(cudaPeekAtLastError());

		if (m_totalAABBUpdateFunc)
		{
			AABB<REAL> totalAABB;
			m_totalAABBBuffer.CopyToHost(&totalAABB, sizeof(AABB<REAL>));
			m_totalAABBUpdateFunc(totalAABB);
		}
	}

	UpdateRenderBuffers();
}

void TriangleBVHTree::ResetTotalAABBBuffer()
{
	AABB<REAL> aabb;
	aabb.Initialize();
	m_totalAABBBuffer.CopyFromHost(&aabb, sizeof(AABB<REAL>));
}

void TriangleBVHTree::BuildRenderBuffers()
{
	if (m_bvhNodes.IsEmpty())
	{
		lineIndices.Clear();
		vertexPositions.Clear();
		vertexOffsets.Clear();
		lineDrawIndirects.Clear();
		return;
	}

	const auto aabbCount = m_bvhNodes.GetBuffer().GetSize() + 1;
	const auto lineCount = aabbCount * 12;
	const auto vertexCount = aabbCount * 8;
	lineIndices.SetSize(lineCount * 2);
	vertexPositions.SetSize(vertexCount);

	const auto treeCount = m_hostBVHTreeInfos.GetSize();
	vertexOffsets.SetSize(treeCount + 2);
	lineDrawIndirects.SetSize(treeCount + 1);

	std::vector<IndexType> offsets;
	std::vector<mvk::DrawIndirectCommand> drawCommands;
	offsets.reserve(treeCount + 2);
	drawCommands.reserve(treeCount + 1);
	for (IndexType i = 0; i < treeCount; ++i)
	{
		offsets.push_back(static_cast<IndexType>(m_bvhNodes.GetRange(i)->offset * 8));

		mvk::DrawIndirectCommand cmd;
		cmd.vertexCount = static_cast<uint32_t>(m_bvhNodes.GetRange(i)->size * 12 * 6);
		cmd.instanceCount = 1;
		cmd.firstVertex = static_cast<uint32_t>(m_bvhNodes.GetRange(i)->offset * 12 * 6);
		cmd.firstInstance = static_cast<uint32_t>(i);
		drawCommands.push_back(cmd);
	}
	{
		const auto lastOffset = static_cast<uint32_t>(m_bvhNodes.GetRanges().back().offset + m_bvhNodes.GetRanges().back().size);
		offsets.push_back(lastOffset * 8);

		mvk::DrawIndirectCommand cmd;
		cmd.vertexCount = static_cast<uint32_t>(12 * 6);
		cmd.instanceCount = 1;
		cmd.firstVertex = lastOffset * 12 * 6;
		cmd.firstInstance = static_cast<uint32_t>(treeCount);
		drawCommands.push_back(cmd);
	}
	offsets.push_back(offsets.back() + 8);
	vertexOffsets.CopyFromHost(offsets.data(), vertexOffsets.GetElementOffset(offsets.size()));
	lineDrawIndirects.CopyFromHost(drawCommands.data(), lineDrawIndirects.GetElementOffset(drawCommands.size()));

	UpdateRenderBuffers();
}

void TriangleBVHTree::UpdateRenderBuffers()
{
	const auto nodeCount = m_bvhNodes.GetBuffer().GetSize();
	if (!nodeCount) return;

	const auto gridCount = mcore::DivUp<BLOCK_ELEMENTS>(nodeCount);
	kernel_UpdateBVHRenderBuffers<BLOCK_SIZE, ELEMENTS_PER_THREAD> << <gridCount, BLOCK_SIZE >> > (
		GetDeviceData(),
		m_totalAABBBuffer.GetData(),
		lineIndices.GetData(),
		vertexPositions.GetData());
	CUDA_CHECK(cudaPeekAtLastError());

	/*{
		std::vector<IndexType> hostLineIndices;
		hostLineIndices.resize(lineIndices.GetSize());
		lineIndices.CopyToHost(hostLineIndices.data(), sizeof(IndexType) * hostLineIndices.size());
		for (IndexType i = 0; i < hostLineIndices.size(); ++i)
		{
			Logger::InfoNoNewLine("Line Index ", i, ": ");
			Logger::Info(hostLineIndices[i]);
		}
		std::vector<IndexType> hostVertexOffset;
		hostVertexOffset.resize(vertexOffsets.GetSize());
		vertexOffsets.CopyToHost(hostVertexOffset.data(), sizeof(IndexType) * hostVertexOffset.size());
		for (IndexType i = 0; i < hostVertexOffset.size(); ++i)
		{
			Logger::InfoNoNewLine("Vertex Offset ", i, ": ");
			Logger::Info(hostVertexOffset[i]);
		}
		Logger::Print();
	}*/
}