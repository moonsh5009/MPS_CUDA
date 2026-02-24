#include "stdafx.h"
#include "MeshUtil.cuh"

namespace
{
	constexpr size_t BLOCK_SIZE = 1024;
	constexpr size_t ELEMENTS_PER_THREAD = 8;
	constexpr size_t BLOCK_ELEMENTS = BLOCK_SIZE * ELEMENTS_PER_THREAD;
}

void MeshUtil::ComputeNormal(const DeviceMeshData& meshData)
{
	const auto nodeCount = meshData.nodeInfo.valueCount;
	const auto faceCount = meshData.faceInfo.valueCount / 3;
	const auto nodeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(nodeCount);
	const auto faceGridCount = mcore::DivUp<BLOCK_ELEMENTS>(faceCount);

	kernel_InitNormal<BLOCK_SIZE, ELEMENTS_PER_THREAD> << <nodeGridCount, BLOCK_SIZE >> > (
		meshData.normals,
		nodeCount);
	CUDA_CHECK(cudaPeekAtLastError());

	kernel_ComputeFaceNormals<BLOCK_SIZE, ELEMENTS_PER_THREAD> << <faceGridCount, BLOCK_SIZE >> > (
		meshData);
	CUDA_CHECK(cudaPeekAtLastError());

	kernel_NormalizeNormals<BLOCK_SIZE, ELEMENTS_PER_THREAD> << <nodeGridCount, BLOCK_SIZE >> > (
		meshData.normals,
		nodeCount);
	CUDA_CHECK(cudaPeekAtLastError());
}

void MeshUtil::UpdateFaceIndirects(const DeviceMeshData& meshData)
{
	const auto meshCount = meshData.info.count;
	const auto meshGridCount = mcore::DivUp<BLOCK_ELEMENTS>(meshCount);

	kernel_UpdateTriangleIndirects<BLOCK_SIZE, ELEMENTS_PER_THREAD> << <meshGridCount, BLOCK_SIZE >> > (
		meshData);
	CUDA_CHECK(cudaPeekAtLastError());
}
