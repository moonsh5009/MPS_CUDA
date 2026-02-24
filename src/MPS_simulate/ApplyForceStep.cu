#include "stdafx.h"
#include "ForceDynamicsSimulator.cuh"

#include "ApplyForceStep.h"

#include <thrust/sequence.h>
#include <thrust/device_ptr.h>

namespace
{
	constexpr size_t BLOCK_SIZE = 1024;
	constexpr size_t ELEMENTS_PER_THREAD = 8;
	constexpr size_t BLOCK_ELEMENTS = BLOCK_SIZE * ELEMENTS_PER_THREAD;
}

REGISTRY_SIMULATE_STEP(ApplyForceStep, 100)

ApplyForceStep::ApplyForceStep(mcore::ISimulateManager* pManager)
	: mcore::simulate::SimulateStep{ pManager }
{}

void ApplyForceStep::Initialize()
{}

void ApplyForceStep::Execute(cudaStream_t stream, REAL dt)
{
	const auto& meshContainer = GetManager()->GetDeviceContainer<DeviceMeshContainer>();
	meshContainer->bvhTree->Refit(
		meshContainer->faceIndices,
		meshContainer->nodeBuffers.GetDeviceInfo(),
		meshContainer->GetNodes());

	const auto& forceDynamicsContainer = GetManager()->GetDeviceContainer<DeviceForceDynamicsContainer>();
	if (forceDynamicsContainer->IsEmpty() || activeNodeIndices.IsEmpty() || activeKineticIndices.IsEmpty())
		return;

	InitForce();
	ApplyGravity();
	IntegrateForce();
}

void ApplyForceStep::OnUpdatedContainer()
{
	UpdateActiveNodeIndices();
}

void ApplyForceStep::UpdateActiveNodeIndices()
{
	const auto& forceDynamicsContainer = GetManager()->GetDeviceContainer<DeviceForceDynamicsContainer>();
	const auto& meshContainer = GetManager()->GetDeviceContainer<DeviceMeshContainer>();
	const auto& kineticContainer = GetManager()->GetDeviceContainer<DeviceKineticContainer>();
	if (forceDynamicsContainer->IsEmpty() || meshContainer->IsEmpty() || kineticContainer->IsEmpty())
		return;

	const auto meshCount = forceDynamicsContainer->meshIndices.GetSize();
	activeNodeIndices.SetSize(forceDynamicsContainer->activeCount);
	activeKineticIndices.SetSize(forceDynamicsContainer->activeCount);
	activeFixeds.SetSize(forceDynamicsContainer->activeCount);

	size_t offset = 0;
	for (size_t i = 0; i < forceDynamicsContainer->meshIndices.GetSize(); ++i)
	{
		const auto meshIndex = forceDynamicsContainer->GetHostMeshIndices()[i];
		const auto kineticIndex = forceDynamicsContainer->GetHostKineticIndices()[i];
		const auto& meshRange = meshContainer->nodeBuffers.GetRange(meshIndex);
		const auto& kineticRange = kineticContainer->buffers.GetRange(kineticIndex);
		if (!meshRange || !kineticRange)
			continue;
		if (meshRange->size != kineticRange->size)
		{
			Logger::Error("ApplyForceStep::UpdateActiveNodeIndices: Mismatched size between mesh and kinetic data.");
			Logger::Print();
			continue;
		}

		const auto nodeCount = meshRange->size;

		thrust::sequence(
			activeNodeIndices.begin() + offset,
			activeNodeIndices.begin() + offset + nodeCount,
			meshRange->offset);
		thrust::sequence(
			activeKineticIndices.begin() + offset,
			activeKineticIndices.begin() + offset + nodeCount,
			kineticRange->offset);

		const auto& fixedRange = kineticContainer->fixeds.GetRange(kineticIndex);
		if (fixedRange && fixedRange->size == nodeCount)
		{
			cudaMemcpy(
				activeFixeds.GetData() + offset,
				kineticContainer->GetFixeds().GetData() + fixedRange->offset,
				nodeCount * sizeof(IndexType),
				cudaMemcpyDeviceToDevice);
		}
		else
		{
			cudaMemset(activeFixeds.GetData() + offset, 0, nodeCount * sizeof(IndexType));
		}

		offset += nodeCount;
	}
}

void ApplyForceStep::InitForce()
{
	const auto& forceDynamicsContainer = GetManager()->GetDeviceContainer<DeviceForceDynamicsContainer>();
	if (forceDynamicsContainer->IsEmpty() || activeKineticIndices.IsEmpty())
		return;

	const auto activeCount = activeKineticIndices.GetSize();
	const auto activeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeCount);
	const auto deviceForceDynamicsData = forceDynamicsContainer->GetDeviceData();

	kernel_InitForce<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<activeGridCount, BLOCK_SIZE>>>(
		deviceForceDynamicsData,
		activeKineticIndices.GetData(),
		activeCount);
	CUDA_CHECK(cudaPeekAtLastError());
}

void ApplyForceStep::ApplyGravity()
{
	const auto& forceDynamicsContainer = GetManager()->GetDeviceContainer<DeviceForceDynamicsContainer>();
	if (forceDynamicsContainer->IsEmpty() || activeKineticIndices.IsEmpty())
		return;

	const auto activeCount = activeKineticIndices.GetSize();
	const auto activeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeCount);
	const auto deviceForceDynamicsData = forceDynamicsContainer->GetDeviceData();

	kernel_ApplyGravity<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<activeGridCount, BLOCK_SIZE>>>(
		deviceForceDynamicsData,
		activeKineticIndices.GetData(),
		activeFixeds.GetData(),
		activeCount);
	CUDA_CHECK(cudaPeekAtLastError());
}

void ApplyForceStep::IntegrateForce()
{
	const auto& forceDynamicsContainer = GetManager()->GetDeviceContainer<DeviceForceDynamicsContainer>();
	if (forceDynamicsContainer->IsEmpty() || activeKineticIndices.IsEmpty())
		return;

	const auto activeCount = activeKineticIndices.GetSize();
	const auto activeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeCount);
	const auto deviceForceDynamicsData = forceDynamicsContainer->GetDeviceData();

	kernel_IntegrateForce<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<activeGridCount, BLOCK_SIZE>>>(
		deviceForceDynamicsData,
		activeKineticIndices.GetData(),
		activeFixeds.GetData(),
		activeCount);
	CUDA_CHECK(cudaPeekAtLastError());
}
