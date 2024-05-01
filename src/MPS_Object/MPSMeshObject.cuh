#pragma once
#include "MPSMeshObject.h"
#include "MPSRTriangleUtil.cuh"

#include "../MCUDA_Lib/MCUDAHelper.cuh"

__global__ void ComputeMeshFaceInfo_kernel(
	const glm::uvec3* MCUDA_RESTRICT pFace, size_t numFaces,
	const REAL3* MCUDA_RESTRICT pPos,
	REAL3* MCUDA_RESTRICT pNorm,
	REAL* MCUDA_RESTRICT pArea)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= numFaces) return;

	const auto face = pFace[id];
	const auto x0 = pPos[face[0]];
	const auto x1 = pPos[face[1]];
	const auto x2 = pPos[face[2]];

	auto norm = glm::cross(x1 - x0, x2 - x0);
	const auto area2 = glm::length(norm);
	pNorm[id] = norm * 1.0 / area2;
	pArea[id] = 0.5 * area2;
}

__global__ void TranslateMesh_kernel(
	REAL3* MCUDA_RESTRICT pPos, size_t numVertices,
	REAL3 oriCenter, REAL3 center, REAL3 scale)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= numVertices) return;

	const auto x = pPos[id];
	pPos[id] = (x - oriCenter) * scale + center;
}

__global__ void RTriangleBuild_kernel(
	const glm::uvec3* MCUDA_RESTRICT pFace, size_t numFaces,
	const uint32_t* MCUDA_RESTRICT nbFs, const uint32_t* MCUDA_RESTRICT inbFs,
	uint32_t* MCUDA_RESTRICT pRTri)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numFaces) return;

	const auto face = pFace[id];

	uint32_t iEnd, jStart, jEnd;
	uint32_t iTri, jTri, ino, jno;
	uint32_t info = 0u;

	jStart = inbFs[face[0]];
	jEnd = inbFs[face[0] + 1u];
#pragma unroll
	for (auto i = 0u, j = 1u; i < 3u; i++, j = (j == 2u ? 0u : j + 1u))
	{
		ino = jStart;
		iEnd = jEnd;
		iTri = nbFs[ino++];
		if (iTri == id)
			mps::SetRTriVertex(info, i);

		jStart = inbFs[face[j]];
		jEnd = inbFs[face[j] + 1u];

		uint32_t of = 0xffffffff;
	#pragma unroll
		for (jno = jStart; jno < jEnd; jno++)
		{
			jTri = nbFs[jno];
			if (jTri == iTri && id != iTri)
			{
				of = iTri;
				break;
			}
		}
	#pragma unroll
		for (; ino < iEnd && of == 0xffffffff; ino++)
		{
			iTri = nbFs[ino];
		#pragma unroll
			for (jno = jStart; jno < jEnd; jno++)
			{
				jTri = nbFs[jno];
				if (jTri == iTri && id != iTri)
				{
					of = iTri;
					break;
				}
			}
		}
		if (id < of)
			mps::SetRTriEdge(info, i);
	}
	pRTri[id] = info;
}

__global__ void ComputeNbFacesSize_kernel(
	const glm::uvec3* MCUDA_RESTRICT pFace, size_t numFaces,
	uint32_t* MCUDA_RESTRICT nbFacesID)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numFaces) return;

	const auto face = pFace[id];
	mcuda::util::AtomicAdd(nbFacesID + face[0], 1u);
	mcuda::util::AtomicAdd(nbFacesID + face[1], 1u);
	mcuda::util::AtomicAdd(nbFacesID + face[2], 1u);
}

__global__ void ComputeNbFaces_kernel(
	const glm::uvec3* MCUDA_RESTRICT pFace, size_t numFaces,
	uint2* MCUDA_RESTRICT vertexID, uint32_t* MCUDA_RESTRICT nbFacesID)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numFaces) return;

	const auto face = pFace[id];
	const auto ino0 = mcuda::util::AtomicAdd(nbFacesID + face[0], 1u);
	const auto ino1 = mcuda::util::AtomicAdd(nbFacesID + face[1], 1u);
	const auto ino2 = mcuda::util::AtomicAdd(nbFacesID + face[2], 1u);
	vertexID[ino0] = make_uint2(face[0], id);
	vertexID[ino1] = make_uint2(face[1], id);
	vertexID[ino2] = make_uint2(face[2], id);
}

__global__ void ComputeNbNodesSize_kernel(
	const glm::uvec3* MCUDA_RESTRICT pFace, size_t numFaces, const uint32_t* MCUDA_RESTRICT pRTri,
	uint32_t* MCUDA_RESTRICT nbNodesID)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numFaces) return;

	const auto face = pFace[id];
	const auto rTri = pRTri[id];
	if (mps::RTriEdge(rTri, 0u))
	{
		mcuda::util::AtomicAdd(nbNodesID + face[0], 1u);
		mcuda::util::AtomicAdd(nbNodesID + face[1], 1u);
	}
	if (mps::RTriEdge(rTri, 1u))
	{
		mcuda::util::AtomicAdd(nbNodesID + face[1], 1u);
		mcuda::util::AtomicAdd(nbNodesID + face[2], 1u);
	}
	if (mps::RTriEdge(rTri, 2u))
	{
		mcuda::util::AtomicAdd(nbNodesID + face[2], 1u);
		mcuda::util::AtomicAdd(nbNodesID + face[0], 1u);
	}
}

__global__ void ComputeNbNodes_kernel(
	const glm::uvec3* MCUDA_RESTRICT pFace, size_t numFaces, const uint32_t* MCUDA_RESTRICT pRTri,
	uint2* MCUDA_RESTRICT vertexID, uint32_t* MCUDA_RESTRICT nbNodesID)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numFaces) return;

	const auto face = pFace[id];
	const auto rTri = pRTri[id];
	if (mps::RTriEdge(rTri, 0u))
	{
		const auto ino0 = mcuda::util::AtomicAdd(nbNodesID + face[0], 1u);
		const auto ino1 = mcuda::util::AtomicAdd(nbNodesID + face[1], 1u);
		vertexID[ino0] = make_uint2(face[0], face[1]);
		vertexID[ino1] = make_uint2(face[1], face[0]);
	}
	if (mps::RTriEdge(rTri, 1u))
	{
		const auto ino0 = mcuda::util::AtomicAdd(nbNodesID + face[1], 1u);
		const auto ino1 = mcuda::util::AtomicAdd(nbNodesID + face[2], 1u);
		vertexID[ino0] = make_uint2(face[1], face[2]);
		vertexID[ino1] = make_uint2(face[2], face[1]);
	}
	if (mps::RTriEdge(rTri, 2u))
	{
		const auto ino0 = mcuda::util::AtomicAdd(nbNodesID + face[2], 1u);
		const auto ino1 = mcuda::util::AtomicAdd(nbNodesID + face[0], 1u);
		vertexID[ino0] = make_uint2(face[2], face[0]);
		vertexID[ino1] = make_uint2(face[0], face[2]);
	}
}