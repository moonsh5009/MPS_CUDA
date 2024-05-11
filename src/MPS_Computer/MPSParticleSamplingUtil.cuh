#pragma once

#include "MPSParticleSamplingUtil.h"

#include "../MPS_Object/MPSMeshParam.h"
#include "../MPS_Object/MPSBoundaryParticleParam.h"
#include "../MPS_Object/MPSMeshMaterial.h"
#include "../MPS_Object/MPSRTriangleUtil.cuh"

namespace mps::device::ParticleSampling
{
	MCUDA_DEVICE_FUNC constexpr uint32_t GetLineSamplingNum(REAL l, REAL d)
	{
		const auto x = (l - d * static_cast<REAL>(1.01)) / d;
		if (x <= static_cast<REAL>(0.0)) return 0u;
		return static_cast<uint32_t>(ceilf(static_cast<float>(x)));
	}

	MCUDA_DEVICE_FUNC void GenerateParticle(
		const REAL3& x, uint32_t faceID,
		const mps::MeshMaterialParam& objMaterial,
		uint32_t* pFaceID,
		REAL2* pBCC,
		REAL3* pPosition,
		REAL* pRadius,
		glm::fvec4* pColor,
		uint32_t& iCurr)
	{
		pFaceID[iCurr] = faceID;
		pBCC[iCurr] = { static_cast<REAL>(10.0), static_cast<REAL>(10.0) };
		pPosition[iCurr] = x;
		pRadius[iCurr] = objMaterial.radius * 0.25;
		pColor[iCurr++] = objMaterial.frontColor;
	}

	MCUDA_DEVICE_FUNC void GenerateLineParticle(
		REAL3 a, REAL3 b, float d, uint32_t faceID,
		const mps::MeshMaterialParam& objMaterial,
		uint32_t* pFaceID,
		REAL2* pBCC,
		REAL3* pPosition,
		REAL* pRadius,
		glm::fvec4* pColor,
		uint32_t& iCurr)
	{
		b -= a;
		const auto l = glm::length(b);
		const auto num = GetLineSamplingNum(l, d);
		if (num == 0u) return;

		d = l / static_cast<REAL>(num + 1u);
		b *= d / l;
		for (auto i = 0u; i < num; i++)
		{
			a += b;
			GenerateParticle(
				a, faceID,
				objMaterial,
				pFaceID,
				pBCC,
				pPosition,
				pRadius,
				pColor,
				iCurr);
		}
	}
}

__global__ void ComputeSamplingNum_kernel(
	mps::MeshMaterialParam objMaterial,
	const glm::uvec3* MCUDA_RESTRICT pObjFace,
	const REAL3* MCUDA_RESTRICT pObjPosition,
	const uint32_t* MCUDA_RESTRICT pObjRTri,
	uint32_t* MCUDA_RESTRICT pObjSamplingParticleSize,
	uint32_t* MCUDA_RESTRICT pObjShortEdgeID,
	size_t objSize,
	uint32_t* MCUDA_RESTRICT pPrevIdx,
	uint32_t* MCUDA_RESTRICT pCurrIdx,
	bool* MCUDA_RESTRICT pIsGenerated,
	bool* MCUDA_RESTRICT isApplied)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= objSize) return;

	const auto face = pObjFace[id];
	REAL3 ns[3] = { pObjPosition[face[0]], pObjPosition[face[1]], pObjPosition[face[2]] };
	REAL3 es[3] = { ns[1] - ns[0], ns[2] - ns[1], ns[0] - ns[2] };
	REAL ls[3] = { glm::length(es[0]), glm::length(es[1]), glm::length(es[2]) };

	auto iShort = 0u;
	if (ls[iShort] > ls[1]) iShort = 1u;
	if (ls[iShort] > ls[2]) iShort = 2u;

	const auto d = (objMaterial.radius + objMaterial.radius) * 0.25;

	// Compute Vertex, Edge
	const auto rTri = pObjRTri[id];
	auto num = 0u;
	for (auto i = 0u; i < 3u; i++)
	{
		if (mps::device::RTri::RTriVertex(rTri, i))
			num++;
		if (mps::device::RTri::RTriEdge(rTri, i))
			num += mps::device::ParticleSampling::GetLineSamplingNum(ls[i], d);
	}

	// Compute Scanline
	const auto ino0 = (iShort + 1u) % 3u;
	const auto ino1 = (iShort + 2u) % 3u;
	ns[ino1] = ns[iShort];

	const auto s = glm::normalize(glm::cross(es[iShort], glm::cross(es[iShort], es[ino1])));
	es[ino1] = -es[ino1];

	const auto l = glm::dot(s, es[ino0]);
	const auto nt = mps::device::ParticleSampling::GetLineSamplingNum(l, d);
	if (nt > 0u)
	{
		const auto stride = (l / static_cast<REAL>(nt + 1u)) / l;
		for (auto i = 0u; i < nt; i++)
		{
			ns[ino0] += stride * es[ino0];
			ns[ino1] += stride * es[ino1];
			num += mps::device::ParticleSampling::GetLineSamplingNum(glm::length(ns[ino0] - ns[ino1]), d);
		}
	}

	pPrevIdx[id] = pObjSamplingParticleSize[id];
	pCurrIdx[id] = num;

	auto isGenerated = false;
	if (iShort != pObjShortEdgeID[id])
	{
		pObjShortEdgeID[id] = iShort;
		isGenerated = true;
	}
	if (num != pObjSamplingParticleSize[id])
	{
		pObjSamplingParticleSize[id] = num;
		isGenerated = true;
	}
	pIsGenerated[id] = isGenerated;
	if (isGenerated) *isApplied = true;
}

__global__ void GenerateBoundaryParticle_kernel(
	mps::MeshMaterialParam objMaterial,
	const glm::uvec3* MCUDA_RESTRICT pObjFace,
	const REAL3* MCUDA_RESTRICT pObjPosition,
	const uint32_t* MCUDA_RESTRICT pObjRTri,
	const uint32_t* MCUDA_RESTRICT pObjShortEdgeID,
	size_t objSize,
	uint32_t* MCUDA_RESTRICT pBoundaryParticleFaceID,
	REAL2* MCUDA_RESTRICT pBoundaryParticleBCC,
	REAL3* MCUDA_RESTRICT pBoundaryParticlePosition,
	REAL* MCUDA_RESTRICT pBoundaryParticleRadius,
	glm::fvec4* MCUDA_RESTRICT pBoundaryParticleColor,
	const uint32_t* MCUDA_RESTRICT pPrevFaceID,
	const REAL2* MCUDA_RESTRICT pPrevBCC,
	const REAL3* MCUDA_RESTRICT pPrevPosition,
	const REAL* MCUDA_RESTRICT pPrevRadius,
	const glm::fvec4* MCUDA_RESTRICT pPrevColor,
	const uint32_t* MCUDA_RESTRICT pPrevIdx,
	const uint32_t* MCUDA_RESTRICT pCurrIdx,
	const bool* MCUDA_RESTRICT pIsGenerated)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= objSize) return;

	auto iCurr = pCurrIdx[id];
	const auto iEnd = pCurrIdx[id + 1u];
	if (pIsGenerated[id])
	{
		const auto face = pObjFace[id];
		REAL3 ns[3] = { pObjPosition[face[0]], pObjPosition[face[1]], pObjPosition[face[2]] };
		REAL3 es[3] = { ns[1] - ns[0], ns[2] - ns[1], ns[0] - ns[2] };
		const auto iShort = pObjShortEdgeID[id];
		const auto lshort = glm::length(es[iShort]);

		const auto d = (objMaterial.radius + objMaterial.radius) * 0.25;

		// Compute Vertex, Edge
		const auto rTri = pObjRTri[id];
		for (auto i = 0u; i < 3u; i++)
		{
			if (mps::device::RTri::RTriVertex(rTri, i))
				mps::device::ParticleSampling::GenerateParticle(ns[i], id,
					objMaterial,
					pBoundaryParticleFaceID,
					pBoundaryParticleBCC,
					pBoundaryParticlePosition,
					pBoundaryParticleRadius,
					pBoundaryParticleColor,
					iCurr);
			if (mps::device::RTri::RTriEdge(rTri, i))
				mps::device::ParticleSampling::GenerateLineParticle(ns[i], ns[(i + 1u) % 3u], d, id,
					objMaterial,
					pBoundaryParticleFaceID,
					pBoundaryParticleBCC,
					pBoundaryParticlePosition,
					pBoundaryParticleRadius,
					pBoundaryParticleColor,
					iCurr);
		}

		// Compute Scanline
		const auto ino0 = (iShort + 1u) % 3u;
		const auto ino1 = (iShort + 2u) % 3u;
		ns[ino1] = ns[iShort];

		const auto s = glm::normalize(glm::cross(es[iShort], glm::cross(es[iShort], es[ino1])));
		es[ino1] = -es[ino1];

		const auto l = glm::dot(s, es[ino0]);
		const auto nt = mps::device::ParticleSampling::GetLineSamplingNum(l, d);
		if (nt > 0u)
		{
			const auto stride = (l / static_cast<REAL>(nt + 1u)) / l;
			for (auto i = 0u; i < nt; i++)
			{
				ns[ino0] += stride * es[ino0];
				ns[ino1] += stride * es[ino1];
				mps::device::ParticleSampling::GenerateLineParticle(ns[ino0], ns[ino1], d, id,
					objMaterial,
					pBoundaryParticleFaceID,
					pBoundaryParticleBCC,
					pBoundaryParticlePosition,
					pBoundaryParticleRadius,
					pBoundaryParticleColor,
					iCurr);
			}
		}
	}
	else
	{
		auto iPrev = pPrevIdx[id];
		while (iCurr < iEnd)
		{
			pBoundaryParticleFaceID[iCurr] = pPrevFaceID[iPrev];
			pBoundaryParticleBCC[iCurr] = pPrevBCC[iPrev];
			pBoundaryParticlePosition[iCurr] = pPrevPosition[iPrev];
			pBoundaryParticleRadius[iCurr] = pPrevRadius[iPrev];
			pBoundaryParticleColor[iCurr++] = pPrevColor[iPrev++];
		}
	}
}

__global__ void SetBarycentric_kernel(
	const uint32_t* MCUDA_RESTRICT pBoundaryParticleFaceID,
	REAL2* MCUDA_RESTRICT pBoundaryParticleBCC,
	const REAL3* MCUDA_RESTRICT pBoundaryParticlePosition,
	REAL* MCUDA_RESTRICT pBoundaryParticleMass,
	size_t boundaryParticleSize,
	const glm::uvec3* MCUDA_RESTRICT pObjFace,
	const REAL3* MCUDA_RESTRICT pObjPosition,
	const REAL* MCUDA_RESTRICT pObjMass)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= boundaryParticleSize) return;

	const auto faceID = pBoundaryParticleFaceID[id];
	const auto face = pObjFace[faceID];

	auto bcc = pBoundaryParticleBCC[id];
	if (bcc[0] == 10.0)
	{
		REAL3 ns[3] = { pObjPosition[face[0]], pObjPosition[face[1]], pObjPosition[face[2]] };
		const auto x = pBoundaryParticlePosition[id];

		auto w0 = static_cast<REAL>(0.0);
		auto w1 = static_cast<REAL>(0.0);
		const auto n20 = ns[0] - ns[2];
		const auto n21 = ns[1] - ns[2];
		const auto t0 = glm::dot(n20, n20);
		const auto t1 = glm::dot(n21, n21);
		const auto t2 = glm::dot(n20, n21);
		const auto t3 = glm::dot(n20, x - ns[2]);
		const auto t4 = glm::dot(n21, x - ns[2]);
		const auto det = t0 * t1 - t2 * t2;
		if (fabs(det) > static_cast<REAL>(1.0e-20))
		{
			const auto invDet = static_cast<REAL>(1.0) / det;
			w0 = (+t1 * t3 - t2 * t4) * invDet;
			w1 = (-t2 * t3 + t0 * t4) * invDet;
		}

		if (w0 < static_cast<REAL>(0.0))		w0 = static_cast<REAL>(0.0);
		else if (w0 > static_cast<REAL>(1.0))	w0 = static_cast<REAL>(1.0);
		if (w1 < static_cast<REAL>(0.0))		w1 = static_cast<REAL>(0.0);
		else if (w1 > static_cast<REAL>(1.0))	w1 = static_cast<REAL>(1.0);
		pBoundaryParticleBCC[id] = { w0, w1 };
		pBoundaryParticleMass[id] = pObjMass[face[0]] * w0 + pObjMass[face[1]] * w1 + pObjMass[face[2]] * (1.0 - w0 - w1);
	}
}

//__global__ void CompNodeWeights_kernel(
//	ClothParam cloth, PoreParticleParam particles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= particles._numParticles)
//		return;
//
//	uint ino = id << 1u;
//	REAL w0 = particles._ws[ino + 0u];
//	REAL w1 = particles._ws[ino + 1u];
//	REAL w2 = 1.0 - w0 - w1;
//
//	ino = particles._inos[id];
//	ino *= 3u;
//	uint ino0 = cloth._fs[ino + 0u];
//	uint ino1 = cloth._fs[ino + 1u];
//	uint ino2 = cloth._fs[ino + 2u];
//
//	atomicAdd_REAL(particles._nodeWeights + ino0, w0);
//	atomicAdd_REAL(particles._nodeWeights + ino1, w1);
//	atomicAdd_REAL(particles._nodeWeights + ino2, w2);
//}

//__global__ void lerpPosition_kernel(
//	ObjParam obj, BoundaryParticleParam boundaryParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= boundaryParticles._numParticles)
//		return;
//
//	uint ino = id << 1u;
//	REAL w0 = boundaryParticles._ws[ino + 0u];
//	REAL w1 = boundaryParticles._ws[ino + 1u];
//
//	ino = boundaryParticles._inos[id];
//	ino *= 3u;
//	uint ino0 = obj._fs[ino + 0u];
//	uint ino1 = obj._fs[ino + 1u];
//	uint ino2 = obj._fs[ino + 2u];
//
//	REAL3 xs[3];
//	ino0 *= 3u; ino1 *= 3u; ino2 *= 3u;
//	xs[0].x = obj._ns[ino0 + 0u]; xs[0].y = obj._ns[ino0 + 1u]; xs[0].z = obj._ns[ino0 + 2u];
//	xs[1].x = obj._ns[ino1 + 0u]; xs[1].y = obj._ns[ino1 + 1u]; xs[1].z = obj._ns[ino1 + 2u];
//	xs[2].x = obj._ns[ino2 + 0u]; xs[2].y = obj._ns[ino2 + 1u]; xs[2].z = obj._ns[ino2 + 2u];
//
//	REAL3 x = xs[0] * w0 + xs[1] * w1 + (1.0 - w0 - w1) * xs[2];
//	ino = id * 3u;
//	boundaryParticles._xs[ino + 0u] = x.x;
//	boundaryParticles._xs[ino + 1u] = x.y;
//	boundaryParticles._xs[ino + 2u] = x.z;
//}
//__global__ void lerpVelocity_kernel(
//	ObjParam obj, BoundaryParticleParam boundaryParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= boundaryParticles._numParticles)
//		return;
//
//	uint ino = id << 1u;
//	REAL w0 = boundaryParticles._ws[ino + 0u];
//	REAL w1 = boundaryParticles._ws[ino + 1u];
//
//	ino = boundaryParticles._inos[id];
//	ino *= 3u;
//	uint ino0 = obj._fs[ino + 0u];
//	uint ino1 = obj._fs[ino + 1u];
//	uint ino2 = obj._fs[ino + 2u];
//
//	REAL3 vs[3];
//	ino0 *= 3u; ino1 *= 3u; ino2 *= 3u;
//	vs[0].x = obj._vs[ino0 + 0u]; vs[0].y = obj._vs[ino0 + 1u]; vs[0].z = obj._vs[ino0 + 2u];
//	vs[1].x = obj._vs[ino1 + 0u]; vs[1].y = obj._vs[ino1 + 1u]; vs[1].z = obj._vs[ino1 + 2u];
//	vs[2].x = obj._vs[ino2 + 0u]; vs[2].y = obj._vs[ino2 + 1u]; vs[2].z = obj._vs[ino2 + 2u];
//
//	REAL3 v = vs[0] * w0 + vs[1] * w1 + (1.0 - w0 - w1) * vs[2];
//	ino = id * 3u;
//	boundaryParticles._vs[ino + 0u] = v.x;
//	boundaryParticles._vs[ino + 1u] = v.y;
//	boundaryParticles._vs[ino + 2u] = v.z;
//}
//__global__ void lerpForce_kernel(
//	ObjParam cloth, PoreParticleParam poreParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	uint ino = id << 1u;
//	REAL w0 = poreParticles._ws[ino + 0u];
//	REAL w1 = poreParticles._ws[ino + 1u];
//	REAL w2 = 1.0 - w0 - w1;
//
//	ino = poreParticles._inos[id];
//
//	ino *= 3u;
//	uint ino0 = cloth._fs[ino + 0u];
//	uint ino1 = cloth._fs[ino + 1u];
//	uint ino2 = cloth._fs[ino + 2u];
//	REAL nodeWeight0 = poreParticles._nodeWeights[ino0];
//	REAL nodeWeight1 = poreParticles._nodeWeights[ino1];
//	REAL nodeWeight2 = poreParticles._nodeWeights[ino2];
//
//	ino = id * 3u;
//	REAL3 force;
//	force.x = poreParticles._forces[ino + 0u];
//	force.y = poreParticles._forces[ino + 1u];
//	force.z = poreParticles._forces[ino + 2u];
//
//	w0 *= w0 / nodeWeight0;
//	w1 *= w1 / nodeWeight1;
//	w2 *= w2 / nodeWeight2;
//	ino0 *= 3u;
//	ino1 *= 3u;
//	ino2 *= 3u;
//	atomicAdd_REAL(cloth._forces + ino0 + 0u, w0 * force.x);
//	atomicAdd_REAL(cloth._forces + ino0 + 1u, w0 * force.y);
//	atomicAdd_REAL(cloth._forces + ino0 + 2u, w0 * force.z);
//	atomicAdd_REAL(cloth._forces + ino1 + 0u, w1 * force.x);
//	atomicAdd_REAL(cloth._forces + ino1 + 1u, w1 * force.y);
//	atomicAdd_REAL(cloth._forces + ino1 + 2u, w1 * force.z);
//	atomicAdd_REAL(cloth._forces + ino2 + 0u, w2 * force.x);
//	atomicAdd_REAL(cloth._forces + ino2 + 1u, w2 * force.y);
//	atomicAdd_REAL(cloth._forces + ino2 + 2u, w2 * force.z);
//}