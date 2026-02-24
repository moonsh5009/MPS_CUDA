#pragma once

#include "../MCore_util/MCudaUtil.cuh"

#include "MeshUtil.h"

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_InitNormal(
	mcore::Vector3* normals,
	size_t numNodes)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(numNodes, [&](size_t tid)
	{
		normals[tid] = { 0.0, 0.0, 0.0 };
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ComputeFaceNormals(
	DeviceMeshData meshData)
{
	const auto faceCount = meshData.faceInfo.valueCount / 3;
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(faceCount, [&](size_t tid)
	{
		const auto baseIdx = tid * 3;
		const auto rangeIndex = meshData.faceInfo.rangeIndexOfValues[baseIdx];
		const auto nodeOffset = meshData.nodeInfo.rangeOffsets[rangeIndex];
		
		const auto idx0 = nodeOffset + meshData.faces[baseIdx];
		const auto idx1 = nodeOffset + meshData.faces[baseIdx + 1];
		const auto idx2 = nodeOffset + meshData.faces[baseIdx + 2];

		const auto p0 = meshData.nodes[idx0];
		const auto p1 = meshData.nodes[idx1];
		const auto p2 = meshData.nodes[idx2];

		const auto edge1 = p1 - p0;
		const auto edge2 = p2 - p0;
		const auto faceNormal = edge1.cross(edge2);

		mcuda::AtomicAdd(&meshData.normals[idx0].x(), faceNormal.x());
		mcuda::AtomicAdd(&meshData.normals[idx0].y(), faceNormal.y());
		mcuda::AtomicAdd(&meshData.normals[idx0].z(), faceNormal.z());

		mcuda::AtomicAdd(&meshData.normals[idx1].x(), faceNormal.x());
		mcuda::AtomicAdd(&meshData.normals[idx1].y(), faceNormal.y());
		mcuda::AtomicAdd(&meshData.normals[idx1].z(), faceNormal.z());

		mcuda::AtomicAdd(&meshData.normals[idx2].x(), faceNormal.x());
		mcuda::AtomicAdd(&meshData.normals[idx2].y(), faceNormal.y());
		mcuda::AtomicAdd(&meshData.normals[idx2].z(), faceNormal.z());
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_NormalizeNormals(
	mcore::Vector3* normals,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		auto& normal = normals[tid];
		const auto length = normal.norm();
		if (length > 1e-10)
			normal *= 1. / length;
		else
			normal = { 0.f, 0.f, 1.0 };
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_UpdateTriangleIndirects(
	DeviceMeshData meshData)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(meshData.info.count, [&](size_t tid)
	{
		const auto faceOffset = meshData.faceInfo.rangeOffsets[tid];
		const auto nextFaceOffset = meshData.faceInfo.rangeOffsets[tid + 1];
		const auto vertexOffset = meshData.nodeInfo.rangeOffsets[tid];

		mvk::DrawIndexedIndirectCommand indirect;
		indirect.indexCount = nextFaceOffset - faceOffset;
		indirect.instanceCount = 1;
		indirect.firstIndex = faceOffset;
		indirect.vertexOffset = vertexOffset;
		indirect.firstInstance = tid;
		meshData.faceDrawIndirects[tid] = indirect;
	});
}