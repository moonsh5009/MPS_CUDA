#pragma once

#include "../MCore_util/MCudaUtil.cuh"

#include "DeviceForceDynamicsContainer.h"

#include "DevicePhysicsConstant.cuh"

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_InitForce(
	DeviceForceDynamicsData forceDynamicsData,
	IndexType* activeKineticIndices,
	size_t activeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(activeCount, [&](size_t tid)
	{
		const auto kineticIndex = activeKineticIndices[tid];
		forceDynamicsData.kineticInfo.forces[kineticIndex] = { 0.0, 0.0, 0.0 };
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_ApplyGravity(
	DeviceForceDynamicsData forceDynamicsData,
	IndexType* activeKineticIndices,
	IndexType* isFixed,
	size_t activeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(activeCount, [&](size_t tid)
	{
		if (isFixed[tid]) return;

		const auto kineticIndex = activeKineticIndices[tid];
		const auto mass = forceDynamicsData.kineticInfo.masses[kineticIndex];
		mcore::Vector3 gravity{ physicsData.gravity.x, physicsData.gravity.y, physicsData.gravity.z };
		forceDynamicsData.kineticInfo.forces[kineticIndex] += gravity * mass;
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_IntegrateForce(
	DeviceForceDynamicsData forceDynamicsData,
	IndexType* activeKineticIndices,
	IndexType* isFixed,
	size_t activeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(activeCount, [&](size_t tid)
	{
		if (isFixed[tid]) return;

		const auto kineticIndex = activeKineticIndices[tid];
		const auto invMass = forceDynamicsData.kineticInfo.invMasses[kineticIndex];
		const auto force = forceDynamicsData.kineticInfo.forces[kineticIndex];
		const auto acceleration = force * invMass;
		forceDynamicsData.kineticInfo.velocities[kineticIndex] += acceleration * physicsData.dt;
	});
}

template<size_t BLOCK_SIZE, size_t ELEMENTS_PER_THREAD>
__global__ void kernel_IntegrateVelocity(
	DeviceForceDynamicsData forceDynamicsData,
	IndexType* activeNodeIndices,
	IndexType* activeKineticIndices,
	IndexType* isFixed,
	size_t activeCount)
{
	mcuda::kernel_Loop<BLOCK_SIZE, ELEMENTS_PER_THREAD>(activeCount, [&](size_t tid)
	{
		if (isFixed[tid]) return;

		const auto nodeIndex = activeNodeIndices[tid];
		const auto kineticIndex = activeKineticIndices[tid];
		const auto velocity = forceDynamicsData.kineticInfo.velocities[kineticIndex];
		forceDynamicsData.meshInfo.nodes[nodeIndex] += velocity * physicsData.dt;
	});
}