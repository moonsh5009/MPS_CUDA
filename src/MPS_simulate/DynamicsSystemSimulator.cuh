#pragma once

#include "../MCore_util/MCudaUtil.cuh"

#include "DeviceDynamicsSystemContainer.h"
#include "DeviceMeshContainer.h"
#include "DeviceKineticContainer.h"

#include "DevicePhysicsConstant.cuh"

// ============================================================
// Vector operations
// ============================================================

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ZeroVector(
	mcore::Vector3* vec,
	size_t count)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(count, [&](size_t tid)
	{
		vec[tid] = { 0.0, 0.0, 0.0 };
	});
}

// ============================================================
// Gravity
// ============================================================

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ApplyGravity(
	mcore::Vector3* velocities,
	IndexType* dynKineticIndices,
	IndexType* isFixed,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		if (isFixed[tid]) return;

		const auto kineticIdx = dynKineticIndices[tid];
		mcore::Vector3 gravity{ physicsData.gravity.x, physicsData.gravity.y, physicsData.gravity.z };
		velocities[kineticIdx] += gravity * physicsData.dt;
	});
}

// ============================================================
// RHS assembly
// ============================================================

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ScaleRHSByDt(
	mcore::Vector3* rhs,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		rhs[tid] = rhs[tid] * physicsData.dt;
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ZeroFixedRHS(
	IndexType* isFixed,
	mcore::Vector3* rhs,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		if (isFixed[tid])
		{
			rhs[tid] = { 0.0, 0.0, 0.0 };
		}
	});
}

// ============================================================
// Preconditioner kernels
// ============================================================

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_InitDiagMass(
	double* masses,
	IndexType* dynKineticIndices,
	mcore::Vector3* diagVec,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		const auto kineticIdx = dynKineticIndices[tid];
		const auto mass = masses[kineticIdx];
		diagVec[tid] = { mass, mass, mass };
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_FinalizePreconditioner(
	IndexType* isFixed,
	mcore::Vector3* diagVec,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		if (isFixed[tid])
		{
			diagVec[tid] = { 1.0, 1.0, 1.0 };
			return;
		}
		auto& dv = diagVec[tid];
		dv.x() = 1.0 / fmax(dv.x(), 1e-10);
		dv.y() = 1.0 / fmax(dv.y(), 1e-10);
		dv.z() = 1.0 / fmax(dv.z(), 1e-10);
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ApplyPreconditioner(
	mcore::Vector3* precond,
	mcore::Vector3* src,
	mcore::Vector3* dst,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		const auto& p = precond[tid];
		const auto& s = src[tid];
		dst[tid] = mcore::Vector3{ p.x() * s.x(), p.y() * s.y(), p.z() * s.z() };
	});
}

// ============================================================
// CG phase kernels
// ============================================================

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_InitPCG(
	mcore::Vector3* deltaV,
	mcore::Vector3* residual,
	mcore::Vector3* zVec,
	mcore::Vector3* dir,
	mcore::Vector3* rhsVec,
	mcore::Vector3* precond,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		deltaV[tid] = { 0.0, 0.0, 0.0 };
		const auto r = rhsVec[tid];
		residual[tid] = r;
		const auto& p = precond[tid];
		const auto zVal = mcore::Vector3{ p.x() * r.x(), p.y() * r.y(), p.z() * r.z() };
		zVec[tid] = zVal;
		dir[tid] = zVal;
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_DotProduct(
	mcore::Vector3* a,
	mcore::Vector3* b,
	double* dotResult,
	size_t count)
{
	__shared__ double sdata[BLOCK_SIZE];
	double localSum = 0.0;

	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(count, [&](size_t tid)
	{
		localSum += a[tid].dot(b[tid]);
	});

	sdata[threadIdx.x] = localSum;
	__syncthreads();
	for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1)
	{
		if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
		__syncthreads();
	}
	if (threadIdx.x < 32) mcuda::WarpSum(sdata, threadIdx.x);
	if (threadIdx.x == 0) mcuda::AtomicAdd(dotResult, sdata[0]);
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ComputeAp_Mass(
	double* masses,
	IndexType* dynKineticIndices,
	mcore::Vector3* dir,
	mcore::Vector3* ApVec,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		const auto kineticIdx = dynKineticIndices[tid];
		const auto mass = masses[kineticIdx];
		ApVec[tid] = dir[tid] * mass;
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ApplyFixedAp(
	IndexType* isFixed,
	mcore::Vector3* dir,
	mcore::Vector3* ApVec,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		if (isFixed[tid]) ApVec[tid] = dir[tid];
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_CGUpdateSolution(
	mcore::Vector3* deltaV,
	mcore::Vector3* residual,
	mcore::Vector3* dir,
	mcore::Vector3* ApVec,
	double alpha,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		deltaV[tid] += dir[tid] * alpha;
		residual[tid] -= ApVec[tid] * alpha;
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_CGUpdateDirection(
	mcore::Vector3* zVec,
	mcore::Vector3* dir,
	double beta,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		dir[tid] = zVec[tid] + dir[tid] * beta;
	});
}

// ============================================================
// Velocity and position update
// ============================================================

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_UpdateVelocityAndPosition(
	mcore::Vector3* meshNodes,
	mcore::Vector3* velocities,
	IndexType* dynNodeIndices,
	IndexType* dynKineticIndices,
	IndexType* isFixed,
	mcore::Vector3* deltaV,
	size_t nodeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(nodeCount, [&](size_t tid)
	{
		if (isFixed[tid]) return;

		const auto nodeIdx = dynNodeIndices[tid];
		const auto kineticIdx = dynKineticIndices[tid];

		velocities[kineticIdx] += deltaV[tid];
		meshNodes[nodeIdx] += velocities[kineticIdx] * physicsData.dt;
	});
}