#pragma once

#include "../MCore_util/MCudaUtil.cuh"

#include "DeviceClothDynamicsContainer.h"
#include "DeviceDynamicsSystemContainer.h"
#include "DeviceMeshContainer.h"

#include "DevicePhysicsConstant.cuh"

// ============================================================
// Gradient kernels (contributor)
// ============================================================

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ComputeStretchGradient(
	mcore::Vector3* meshNodes,
	IndexType* dynNodeIndices,
	IndexType* activeEdgeNodeIndices,
	double* activeRestLengths,
	double stretchK,
	mcore::Vector3* rhs,
	size_t edgeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(edgeCount, [&](size_t tid)
	{
		const auto localV0 = activeEdgeNodeIndices[tid * 2 + 0];
		const auto localV1 = activeEdgeNodeIndices[tid * 2 + 1];
		const auto nodeIdx0 = dynNodeIndices[localV0];
		const auto nodeIdx1 = dynNodeIndices[localV1];

		const auto x0 = meshNodes[nodeIdx0];
		const auto x1 = meshNodes[nodeIdx1];

		const auto d = x1 - x0;
		const auto l = d.norm();
		if (l < 1e-12) return;

		const auto L0 = activeRestLengths[tid];
		const auto n = d / l;
		const auto f = n * (-stretchK * (l - L0));

		mcuda::AtomicAdd(&rhs[localV0].x(), -f.x());
		mcuda::AtomicAdd(&rhs[localV0].y(), -f.y());
		mcuda::AtomicAdd(&rhs[localV0].z(), -f.z());
		mcuda::AtomicAdd(&rhs[localV1].x(), f.x());
		mcuda::AtomicAdd(&rhs[localV1].y(), f.y());
		mcuda::AtomicAdd(&rhs[localV1].z(), f.z());
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ComputeBendGradient(
	mcore::Vector3* meshNodes,
	IndexType* dynNodeIndices,
	BendEdge* activeBendEdges,
	double bendK,
	mcore::Vector3* rhs,
	size_t bendEdgeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(bendEdgeCount, [&](size_t tid)
	{
		const auto& be = activeBendEdges[tid];
		const auto nodeIdx0 = dynNodeIndices[be.v0];
		const auto nodeIdx1 = dynNodeIndices[be.v1];
		const auto nodeIdx2 = dynNodeIndices[be.v2];
		const auto nodeIdx3 = dynNodeIndices[be.v3];

		const auto x0 = meshNodes[nodeIdx0];
		const auto x1 = meshNodes[nodeIdx1];
		const auto x2 = meshNodes[nodeIdx2];
		const auto x3 = meshNodes[nodeIdx3];

		const auto e = x1 - x0;
		const auto eLen = e.norm();
		if (eLen < 1e-12) return;

		const auto e0 = x2 - x0;
		const auto e1 = x3 - x0;

		const mcore::Vector3 n0 = e.cross(e0);
		const mcore::Vector3 n1 = e.cross(e1);

		const auto n0Len = n0.norm();
		const auto n1Len = n1.norm();
		if (n0Len < 1e-12 || n1Len < 1e-12) return;

		const auto n0Hat = n0 / n0Len;
		const auto n1Hat = n1 / n1Len;

		auto cosTheta = n0Hat.dot(n1Hat);
		cosTheta = mcuda::clamp(cosTheta, -1.0, 1.0);

		const auto crossN = n0Hat.cross(n1Hat);
		auto sinTheta = crossN.dot(e / eLen);

		const auto theta = atan2(sinTheta, cosTheta);
		const auto gradScale = -bendK * 2.0 * theta;

		const auto dn0 = (n0Hat * cosTheta - n1Hat) / n0Len;
		const auto dn1 = (n1Hat * cosTheta - n0Hat) / n1Len;

		const auto g2 = e.cross(dn0) * gradScale;
		const auto g3 = e.cross(dn1) * gradScale;
		const auto g0 = (e0.cross(dn0) + e1.cross(dn1)) * (-gradScale);
		const auto g1 = g0 * (-1.0) - g2 - g3;

		mcuda::AtomicAdd(&rhs[be.v0].x(), g0.x());
		mcuda::AtomicAdd(&rhs[be.v0].y(), g0.y());
		mcuda::AtomicAdd(&rhs[be.v0].z(), g0.z());
		mcuda::AtomicAdd(&rhs[be.v1].x(), g1.x());
		mcuda::AtomicAdd(&rhs[be.v1].y(), g1.y());
		mcuda::AtomicAdd(&rhs[be.v1].z(), g1.z());
		mcuda::AtomicAdd(&rhs[be.v2].x(), g2.x());
		mcuda::AtomicAdd(&rhs[be.v2].y(), g2.y());
		mcuda::AtomicAdd(&rhs[be.v2].z(), g2.z());
		mcuda::AtomicAdd(&rhs[be.v3].x(), g3.x());
		mcuda::AtomicAdd(&rhs[be.v3].y(), g3.y());
		mcuda::AtomicAdd(&rhs[be.v3].z(), g3.z());
	});
}

// ============================================================
// Hessian-vector product kernels (contributor)
// ============================================================

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ComputeAp_StretchHv(
	mcore::Vector3* meshNodes,
	IndexType* dynNodeIndices,
	IndexType* activeEdgeNodeIndices,
	double* activeRestLengths,
	double stretchK,
	const mcore::Vector3* dir,
	mcore::Vector3* ApVec,
	size_t edgeCount)
{
	const double dt2 = physicsData.dt * physicsData.dt;

	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(edgeCount, [&](size_t eid)
	{
		const auto localV0 = activeEdgeNodeIndices[eid * 2 + 0];
		const auto localV1 = activeEdgeNodeIndices[eid * 2 + 1];
		const auto nodeIdx0 = dynNodeIndices[localV0];
		const auto nodeIdx1 = dynNodeIndices[localV1];

		const auto x0 = meshNodes[nodeIdx0];
		const auto x1 = meshNodes[nodeIdx1];

		const auto d = x1 - x0;
		const auto l = d.norm();
		if (l < 1e-12) return;

		const auto L0 = activeRestLengths[eid];
		const auto n = d / l;

		const auto p0 = dir[localV0];
		const auto p1 = dir[localV1];
		const auto dp = p1 - p0;

		const auto ndp = n.dot(dp);
		const auto Hp = (dp * (L0 / l - 1.0) - n * ndp * (L0 / l)) * stretchK;

		const auto contribution = Hp * dt2;
		mcuda::AtomicAdd(&ApVec[localV0].x(), contribution.x());
		mcuda::AtomicAdd(&ApVec[localV0].y(), contribution.y());
		mcuda::AtomicAdd(&ApVec[localV0].z(), contribution.z());
		mcuda::AtomicAdd(&ApVec[localV1].x(), -contribution.x());
		mcuda::AtomicAdd(&ApVec[localV1].y(), -contribution.y());
		mcuda::AtomicAdd(&ApVec[localV1].z(), -contribution.z());
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ComputeAp_BendHv(
	mcore::Vector3* meshNodes,
	IndexType* dynNodeIndices,
	BendEdge* activeBendEdges,
	double bendK,
	const mcore::Vector3* dir,
	mcore::Vector3* ApVec,
	size_t bendEdgeCount)
{
	const double dt2 = physicsData.dt * physicsData.dt;
	constexpr double eps = 1e-6;

	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(bendEdgeCount, [&](size_t bid)
	{
		const auto& be = activeBendEdges[bid];
		const auto nodeIdx0 = dynNodeIndices[be.v0];
		const auto nodeIdx1 = dynNodeIndices[be.v1];
		const auto nodeIdx2 = dynNodeIndices[be.v2];
		const auto nodeIdx3 = dynNodeIndices[be.v3];

		const auto x0 = meshNodes[nodeIdx0];
		const auto x1 = meshNodes[nodeIdx1];
		const auto x2 = meshNodes[nodeIdx2];
		const auto x3 = meshNodes[nodeIdx3];

		const auto p0 = dir[be.v0];
		const auto p1 = dir[be.v1];
		const auto p2 = dir[be.v2];
		const auto p3 = dir[be.v3];

		auto computeBendGrad = [&](
			const mcore::Vector3& q0, const mcore::Vector3& q1,
			const mcore::Vector3& q2, const mcore::Vector3& q3,
			mcore::Vector3& g0, mcore::Vector3& g1,
			mcore::Vector3& g2, mcore::Vector3& g3)
		{
			const auto ev = q1 - q0;
			const auto evLen = ev.norm();
			if (evLen < 1e-12) { g0 = g1 = g2 = g3 = mcore::Vector3{ 0,0,0 }; return; }

			const auto e0v = q2 - q0;
			const auto e1v = q3 - q0;
			const mcore::Vector3 fn0 = ev.cross(e0v);
			const mcore::Vector3 fn1 = ev.cross(e1v);
			const auto fn0L = fn0.norm();
			const auto fn1L = fn1.norm();
			if (fn0L < 1e-12 || fn1L < 1e-12) { g0 = g1 = g2 = g3 = mcore::Vector3{ 0,0,0 }; return; }

			const auto fn0h = fn0 / fn0L;
			const auto fn1h = fn1 / fn1L;
			auto cT = fn0h.dot(fn1h);
			cT = mcuda::clamp(cT, -1.0, 1.0);
			const auto cr = fn0h.cross(fn1h);
			const auto sT = cr.dot(ev / evLen);
			const auto theta = atan2(sT, cT);
			const auto gs = -bendK * 2.0 * theta;

			const auto dfn0 = (fn0h * cT - fn1h) / fn0L;
			const auto dfn1 = (fn1h * cT - fn0h) / fn1L;

			g2 = ev.cross(dfn0) * gs;
			g3 = ev.cross(dfn1) * gs;
			g0 = (e0v.cross(dfn0) + e1v.cross(dfn1)) * (-gs);
			g1 = g0 * (-1.0) - g2 - g3;
		};

		mcore::Vector3 g0a, g1a, g2a, g3a;
		computeBendGrad(x0, x1, x2, x3, g0a, g1a, g2a, g3a);

		mcore::Vector3 g0b, g1b, g2b, g3b;
		computeBendGrad(
			x0 + p0 * eps, x1 + p1 * eps,
			x2 + p2 * eps, x3 + p3 * eps,
			g0b, g1b, g2b, g3b);

		const auto scale = -dt2 / eps;
		const auto Hp0 = (g0b - g0a) * scale;
		const auto Hp1 = (g1b - g1a) * scale;
		const auto Hp2 = (g2b - g2a) * scale;
		const auto Hp3 = (g3b - g3a) * scale;

		mcuda::AtomicAdd(&ApVec[be.v0].x(), Hp0.x());
		mcuda::AtomicAdd(&ApVec[be.v0].y(), Hp0.y());
		mcuda::AtomicAdd(&ApVec[be.v0].z(), Hp0.z());
		mcuda::AtomicAdd(&ApVec[be.v1].x(), Hp1.x());
		mcuda::AtomicAdd(&ApVec[be.v1].y(), Hp1.y());
		mcuda::AtomicAdd(&ApVec[be.v1].z(), Hp1.z());
		mcuda::AtomicAdd(&ApVec[be.v2].x(), Hp2.x());
		mcuda::AtomicAdd(&ApVec[be.v2].y(), Hp2.y());
		mcuda::AtomicAdd(&ApVec[be.v2].z(), Hp2.z());
		mcuda::AtomicAdd(&ApVec[be.v3].x(), Hp3.x());
		mcuda::AtomicAdd(&ApVec[be.v3].y(), Hp3.y());
		mcuda::AtomicAdd(&ApVec[be.v3].z(), Hp3.z());
	});
}

// ============================================================
// Diagonal accumulation (contributor)
// ============================================================

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_AccumDiag_Stretch(
	mcore::Vector3* meshNodes,
	IndexType* dynNodeIndices,
	IndexType* activeEdgeNodeIndices,
	double* activeRestLengths,
	double stretchK,
	mcore::Vector3* diagVec,
	size_t edgeCount)
{
	const double dt2 = physicsData.dt * physicsData.dt;

	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(edgeCount, [&](size_t tid)
	{
		const auto localV0 = activeEdgeNodeIndices[tid * 2 + 0];
		const auto localV1 = activeEdgeNodeIndices[tid * 2 + 1];
		const auto nodeIdx0 = dynNodeIndices[localV0];
		const auto nodeIdx1 = dynNodeIndices[localV1];

		const auto x0 = meshNodes[nodeIdx0];
		const auto x1 = meshNodes[nodeIdx1];

		const auto d = x1 - x0;
		const auto l = d.norm();
		if (l < 1e-12) return;

		const auto L0 = activeRestLengths[tid];
		const auto n = d / l;

		const auto factor = 1.0 - L0 / l;
		const auto nFactor = L0 / l;
		const auto dx = dt2 * stretchK * (factor + nFactor * n.x() * n.x());
		const auto dy = dt2 * stretchK * (factor + nFactor * n.y() * n.y());
		const auto dz = dt2 * stretchK * (factor + nFactor * n.z() * n.z());

		mcuda::AtomicAdd(&diagVec[localV0].x(), dx);
		mcuda::AtomicAdd(&diagVec[localV0].y(), dy);
		mcuda::AtomicAdd(&diagVec[localV0].z(), dz);
		mcuda::AtomicAdd(&diagVec[localV1].x(), dx);
		mcuda::AtomicAdd(&diagVec[localV1].y(), dy);
		mcuda::AtomicAdd(&diagVec[localV1].z(), dz);
	});
}

// ============================================================
// Damping kernel (contributor)
// ============================================================

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_AccumDamping(
	mcore::Vector3* velocities,
	IndexType* dynKineticIndices,
	double dampingCoeff,
	mcore::Vector3* rhs,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		const auto kineticIdx = dynKineticIndices[tid];
		const auto velocity = velocities[kineticIdx];
		mcuda::AtomicAdd(&rhs[tid].x(), -velocity.x() * dampingCoeff);
		mcuda::AtomicAdd(&rhs[tid].y(), -velocity.y() * dampingCoeff);
		mcuda::AtomicAdd(&rhs[tid].z(), -velocity.z() * dampingCoeff);
	});
}