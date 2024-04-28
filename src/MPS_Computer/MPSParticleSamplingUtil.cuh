#pragma once
#include "MPSParticleSamplingUtil.h"

#include "../MPS_Object/MPSMeshParam.h"
#include "../MPS_Object/MPSBoundaryParticleObject.h"
#include "../MPS_Object/MPSMeshMaterial.h"
#include "../MPS_Object/MPSRTriangleUtil.cuh"

__device__ uint32_t GetLineSamplingNum(REAL l, REAL d)
{
	const auto x = (l - d * static_cast<REAL>(1.01)) / d;
	if (x <= static_cast<REAL>(0.0)) return 0u;
	return static_cast<uint32_t>(ceilf(static_cast<float>(x)));
}

__device__ void GenerateParticle(const REAL3& x, uint32_t faceID, const mps::MeshMaterialParam& objMaterial, mps::BoundaryParticleParam& boundaryParticle, uint32_t& iCurr)
{
	boundaryParticle.Position(iCurr) = x;
	boundaryParticle.BCC(iCurr) = { static_cast<REAL>(10.0), static_cast<REAL>(10.0) };
	boundaryParticle.FaceID(iCurr) = faceID;
	boundaryParticle.Radius(iCurr) = objMaterial.radius * 0.25;
	boundaryParticle.Color(iCurr++) = objMaterial.frontColor;
}

__device__ void GenerateLineParticle(REAL3 a, REAL3 b, float d, uint32_t faceID, const mps::MeshMaterialParam& objMaterial, mps::BoundaryParticleParam& boundaryParticle, uint32_t& iCurr)
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
		GenerateParticle(a, faceID, objMaterial, boundaryParticle, iCurr);
	}
}

__global__ void ComputeSamplingNum_kernel(
	mps::MeshParam obj, mps::MeshMaterialParam objMaterial,
	uint32_t* MCUDA_RESTRICT prevIdx, uint32_t* MCUDA_RESTRICT currIdx,
	bool* MCUDA_RESTRICT isGenerateds, bool* MCUDA_RESTRICT isApplied)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= obj.GetSize()) return;

	const auto face = obj.Face(id);
	REAL3 ns[3] = { obj.Position(face[0]), obj.Position(face[1]), obj.Position(face[2]) };
	REAL3 es[3] = { ns[1] - ns[0], ns[2] - ns[1], ns[0] - ns[2] };
	REAL ls[3] = { glm::length(es[0]), glm::length(es[1]), glm::length(es[2]) };

	auto iShort = 0u;
	if (ls[iShort] > ls[1]) iShort = 1u;
	if (ls[iShort] > ls[2]) iShort = 2u;

	const auto d = (objMaterial.radius + objMaterial.radius) * 0.25;

	// Compute Vertex, Edge
	const auto rTri = obj.RTriangle(id);
	auto num = 0u;
	for (auto i = 0u; i < 3u; i++)
	{
		if (mps::RTriVertex(rTri, i))
			num++;
		if (mps::RTriEdge(rTri, i))
			num += GetLineSamplingNum(ls[i], d);
	}

	// Compute Scanline
	const auto ino0 = (iShort + 1u) % 3u;
	const auto ino1 = (iShort + 2u) % 3u;
	ns[ino1] = ns[iShort];

	const auto s = glm::normalize(glm::cross(es[iShort], glm::cross(es[iShort], es[ino1])));
	es[ino1] = -es[ino1];

	const auto l = glm::dot(s, es[ino0]);
	const auto nt = GetLineSamplingNum(l, d);
	if (nt > 0u)
	{
		const auto stride = (l / static_cast<REAL>(nt + 1u)) / l;
		for (auto i = 0u; i < nt; i++)
		{
			ns[ino0] += stride * es[ino0];
			ns[ino1] += stride * es[ino1];
			num += GetLineSamplingNum(glm::length(ns[ino0] - ns[ino1]), d);
		}
	}

	prevIdx[id] = obj.SamplingParticleSize(id);
	currIdx[id] = num;

	auto isGenerated = false;
	if (iShort != obj.ShortEdgeId(id))
	{
		obj.ShortEdgeId(id) = iShort;
		isGenerated = true;
	}
	if (num != obj.SamplingParticleSize(id))
	{
		obj.SamplingParticleSize(id) = num;
		isGenerated = true;
	}
	isGenerateds[id] = isGenerated;
	if (isGenerated) *isApplied = true;
}

__global__ void GenerateBoundaryParticle_kernel(
	mps::MeshParam obj, mps::MeshMaterialParam objMaterial, mps::BoundaryParticleParam boundaryParticle,
	REAL3* MCUDA_RESTRICT prevX, REAL2* MCUDA_RESTRICT prevBCC, uint32_t* MCUDA_RESTRICT prevFaceID,
	uint32_t* MCUDA_RESTRICT prevIdx, uint32_t* MCUDA_RESTRICT currIdx, bool* MCUDA_RESTRICT isGenerateds)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= obj.GetSize()) return;

	auto iCurr = currIdx[id];
	const auto iEnd = currIdx[id + 1u];
	if (isGenerateds[id])
	{
		const auto face = obj.Face(id);

		REAL3 ns[3] = { obj.Position(face[0]), obj.Position(face[1]), obj.Position(face[2]) };
		REAL3 es[3] = { ns[1] - ns[0], ns[2] - ns[1], ns[0] - ns[2] };
		const auto iShort = obj.ShortEdgeId(id);
		const auto lshort = glm::length(es[iShort]);

		const auto d = (objMaterial.radius + objMaterial.radius) * 0.25;

		// Compute Vertex, Edge
		const auto rTri = obj.RTriangle(id);
		for (auto i = 0u; i < 3u; i++)
		{
			if (mps::RTriVertex(rTri, i))
				GenerateParticle(ns[i], id, objMaterial, boundaryParticle, iCurr);
			if (mps::RTriEdge(rTri, i))
				GenerateLineParticle(ns[i], ns[(i + 1u) % 3u], d, id, objMaterial, boundaryParticle, iCurr);
		}

		// Compute Scanline
		const auto ino0 = (iShort + 1u) % 3u;
		const auto ino1 = (iShort + 2u) % 3u;
		ns[ino1] = ns[iShort];

		const auto s = glm::normalize(glm::cross(es[iShort], glm::cross(es[iShort], es[ino1])));
		es[ino1] = -es[ino1];

		const auto l = glm::dot(s, es[ino0]);
		const auto nt = GetLineSamplingNum(l, d);
		if (nt > 0u)
		{
			const auto stride = (l / static_cast<REAL>(nt + 1u)) / l;
			for (auto i = 0u; i < nt; i++)
			{
				ns[ino0] += stride * es[ino0];
				ns[ino1] += stride * es[ino1];
				GenerateLineParticle(ns[ino0], ns[ino1], d, id, objMaterial, boundaryParticle, iCurr);
			}
		}
	}
	else
	{
		auto iPrev = prevIdx[id];
		while (iCurr < iEnd)
		{
			boundaryParticle.Position(iCurr) = prevX[iPrev];
			boundaryParticle.BCC(iCurr) = prevBCC[iPrev];
			boundaryParticle.FaceID(iCurr++) = prevFaceID[iPrev++];
		}
	}
}

__global__ void SetBarycentric_kernel(mps::MeshParam obj, mps::MeshMaterialParam objMaterial, mps::BoundaryParticleParam boundaryParticle)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= boundaryParticle.GetSize()) return;

	const auto faceID = boundaryParticle.FaceID(id);
	const auto face = obj.Face(faceID);

	auto bcc = boundaryParticle.BCC(id);
	if (bcc[0] == 10.0)
	{
		REAL3 ns[3] = { obj.Position(face[0]), obj.Position(face[1]), obj.Position(face[2]) };
		const auto x = boundaryParticle.Position(id);

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
		boundaryParticle.BCC(id) = { w0, w1 };
		boundaryParticle.Mass(id) = obj.Mass(face[0]) * w0 + obj.Mass(face[1]) * w1 + obj.Mass(face[2]) * (1.0 - w0 - w1);
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