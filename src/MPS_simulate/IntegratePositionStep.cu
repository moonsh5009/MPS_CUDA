#include "stdafx.h"
#include "ForceDynamicsSimulator.cuh"

#include "IntegratePositionStep.h"
#include "ApplyForceStep.h"

namespace
{
	constexpr size_t BLOCK_SIZE = 1024;
	constexpr size_t ELEMENTS_PER_THREAD = 8;
	constexpr size_t BLOCK_ELEMENTS = BLOCK_SIZE * ELEMENTS_PER_THREAD;
}

REGISTRY_SIMULATE_STEP(IntegratePositionStep, 300)

IntegratePositionStep::IntegratePositionStep(mcore::ISimulateManager* pManager)
	: mcore::simulate::SimulateStep{ pManager }
{}

void IntegratePositionStep::Initialize()
{}

void IntegratePositionStep::PostInitialize()
{
	m_pApplyForceStep = GetManager()->FindStep<ApplyForceStep>();
}

void IntegratePositionStep::Execute(cudaStream_t stream, REAL dt)
{
	if (!m_pApplyForceStep)
		return;

	const auto& forceDynamicsContainer = GetManager()->GetDeviceContainer<DeviceForceDynamicsContainer>();
	if (forceDynamicsContainer->IsEmpty() ||
		m_pApplyForceStep->activeNodeIndices.IsEmpty() ||
		m_pApplyForceStep->activeKineticIndices.IsEmpty())
		return;

	IntegrateVelocity();
}

void IntegratePositionStep::IntegrateVelocity()
{
	const auto& forceDynamicsContainer = GetManager()->GetDeviceContainer<DeviceForceDynamicsContainer>();
	if (forceDynamicsContainer->IsEmpty() ||
		m_pApplyForceStep->activeNodeIndices.IsEmpty() ||
		m_pApplyForceStep->activeKineticIndices.IsEmpty())
		return;

	const auto activeCount = m_pApplyForceStep->activeNodeIndices.GetSize();
	const auto activeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeCount);
	const auto deviceForceDynamicsData = forceDynamicsContainer->GetDeviceData();

	kernel_IntegrateVelocity<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<activeGridCount, BLOCK_SIZE>>>(
		deviceForceDynamicsData,
		m_pApplyForceStep->activeNodeIndices.GetData(),
		m_pApplyForceStep->activeKineticIndices.GetData(),
		m_pApplyForceStep->activeFixeds.GetData(),
		activeCount);
	CUDA_CHECK(cudaPeekAtLastError());
}
