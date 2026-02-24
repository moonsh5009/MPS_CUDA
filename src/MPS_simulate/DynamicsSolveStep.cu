#include "stdafx.h"
#include "DynamicsSystemSimulator.cuh"

#include "DynamicsSolveStep.h"

#include <thrust/sequence.h>
#include <thrust/device_ptr.h>

namespace
{
	constexpr size_t BLOCK_SIZE = 1024;
	constexpr size_t ELEMENTS_PER_THREAD = 8;
	constexpr size_t BLOCK_ELEMENTS = BLOCK_SIZE * ELEMENTS_PER_THREAD;

	constexpr size_t DOT_BLOCK_SIZE = 256;
	constexpr size_t DOT_EPT = 8;
	constexpr size_t DOT_BLOCK_ELEMENTS = DOT_BLOCK_SIZE * DOT_EPT;
}

REGISTRY_SIMULATE_STEP(DynamicsSolveStep, 200)

DynamicsSolveStep::DynamicsSolveStep(mcore::ISimulateManager* pManager)
	: mcore::simulate::SimulateStep{ pManager }
{}

void DynamicsSolveStep::Initialize()
{
	globalDotBuffer.SetSize(1);
}

void DynamicsSolveStep::PostInitialize()
{
	for (const auto& group : GetManager()->GetStepGroups())
		for (const auto& pStep : group)
			if (auto* p = dynamic_cast<mcore::IDynamicsContributor*>(pStep.get()))
				RegisterContributor(p);
}

void DynamicsSolveStep::Execute(cudaStream_t stream, REAL dt)
{
	const auto& dynSysContainer = GetManager()->GetDeviceContainer<DeviceDynamicsSystemContainer>();
	if (dynSysContainer->IsEmpty() || activeNodeCount == 0)
		return;

	ApplyGravity();
	ComputeElasticGradient();
	CGSolve();
	UpdateVelocityAndPosition();
}

void DynamicsSolveStep::OnUpdatedContainer()
{
	UpdateActiveIndices();
}

void DynamicsSolveStep::RegisterContributor(mcore::IDynamicsContributor* pContributor)
{
	m_contributors.push_back(pContributor);
}

mcore::DynamicsContext DynamicsSolveStep::BuildContext() const
{
	const auto& dynSysContainer = GetManager()->GetDeviceContainer<DeviceDynamicsSystemContainer>();
	const auto deviceData = dynSysContainer->GetDeviceData();

	mcore::DynamicsContext ctx;
	ctx.meshNodes = deviceData.meshInfo.nodes;
	ctx.velocities = deviceData.kineticInfo.velocities;
	ctx.masses = deviceData.kineticInfo.masses;
	ctx.dynNodeIndices = dynNodeIndices.GetData();
	ctx.dynKineticIndices = dynKineticIndices.GetData();
	ctx.isFixed = isFixed.GetData();
	ctx.dt = 0.01;
	ctx.totalNodeCount = activeNodeCount;
	return ctx;
}

void DynamicsSolveStep::UpdateActiveIndices()
{
	const auto& dynSysContainer = GetManager()->GetDeviceContainer<DeviceDynamicsSystemContainer>();
	const auto& meshContainer = GetManager()->GetDeviceContainer<DeviceMeshContainer>();
	const auto& kineticContainer = GetManager()->GetDeviceContainer<DeviceKineticContainer>();
	if (dynSysContainer->IsEmpty() || meshContainer->IsEmpty() || kineticContainer->IsEmpty())
		return;

	activeNodeCount = dynSysContainer->totalNodeCount;

	if (activeNodeCount == 0)
		return;

	dynNodeIndices.SetSize(activeNodeCount);
	dynKineticIndices.SetSize(activeNodeCount);
	isFixed.SetSize(activeNodeCount);

	deltaV.SetSize(activeNodeCount);
	rhs.SetSize(activeNodeCount);
	residual.SetSize(activeNodeCount);
	direction.SetSize(activeNodeCount);
	Ap.SetSize(activeNodeCount);
	preconditioner.SetSize(activeNodeCount);
	z.SetSize(activeNodeCount);

	size_t nodeOffset = 0;
	maxIterations = 30;
	tolerance = 1e-8;

	for (size_t i = 0; i < dynSysContainer->meshIndices.GetSize(); ++i)
	{
		const auto meshIndex = dynSysContainer->GetHostMeshIndices()[i];
		const auto kineticIndex = dynSysContainer->GetHostKineticIndices()[i];
		const auto& meshRange = meshContainer->nodeBuffers.GetRange(meshIndex);
		const auto& kineticRange = kineticContainer->buffers.GetRange(kineticIndex);
		if (!meshRange || !kineticRange)
			continue;
		if (meshRange->size != kineticRange->size)
			continue;

		const auto nodeCount = meshRange->size;

		thrust::sequence(
			dynNodeIndices.begin() + nodeOffset,
			dynNodeIndices.begin() + nodeOffset + nodeCount,
			meshRange->offset);
		thrust::sequence(
			dynKineticIndices.begin() + nodeOffset,
			dynKineticIndices.begin() + nodeOffset + nodeCount,
			kineticRange->offset);

		const auto& fixedRange = kineticContainer->fixeds.GetRange(kineticIndex);
		if (fixedRange && fixedRange->size == nodeCount)
		{
			cudaMemcpy(
				isFixed.GetData() + nodeOffset,
				kineticContainer->GetFixeds().GetData() + fixedRange->offset,
				nodeCount * sizeof(IndexType),
				cudaMemcpyDeviceToDevice);
		}
		else
		{
			cudaMemset(isFixed.GetData() + nodeOffset, 0, nodeCount * sizeof(IndexType));
		}

		const auto iterVal = dynSysContainer->maxIterationsArray[static_cast<IndexType>(i)];
		if (iterVal > maxIterations)
			maxIterations = iterVal;

		const auto tolVal = dynSysContainer->toleranceArray[static_cast<IndexType>(i)];
		if (tolVal < tolerance)
			tolerance = tolVal;

		nodeOffset += nodeCount;
	}
}

void DynamicsSolveStep::ApplyGravity()
{
	const auto& dynSysContainer = GetManager()->GetDeviceContainer<DeviceDynamicsSystemContainer>();
	const auto deviceData = dynSysContainer->GetDeviceData();

	const auto nodeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeNodeCount);
	kernel_ApplyGravity<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
		deviceData.kineticInfo.velocities,
		dynKineticIndices.GetData(),
		isFixed.GetData(),
		activeNodeCount);
	CUDA_CHECK(cudaPeekAtLastError());
}

void DynamicsSolveStep::ComputeElasticGradient()
{
	const auto nodeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeNodeCount);

	kernel_ZeroVector<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
		rhs.GetData(), activeNodeCount);
	CUDA_CHECK(cudaPeekAtLastError());

	const auto ctx = BuildContext();

	for (auto* pContributor : m_contributors)
	{
		pContributor->AccumulateGradient(ctx, rhs.GetData());
	}

	kernel_ScaleRHSByDt<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
		rhs.GetData(), activeNodeCount);
	CUDA_CHECK(cudaPeekAtLastError());

	for (auto* pContributor : m_contributors)
	{
		pContributor->AccumulateDamping(ctx, rhs.GetData());
	}

	kernel_ZeroFixedRHS<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
		isFixed.GetData(), rhs.GetData(), activeNodeCount);
	CUDA_CHECK(cudaPeekAtLastError());
}

void DynamicsSolveStep::ComputePreconditioner()
{
	const auto& dynSysContainer = GetManager()->GetDeviceContainer<DeviceDynamicsSystemContainer>();
	const auto deviceData = dynSysContainer->GetDeviceData();

	const auto nodeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeNodeCount);

	kernel_InitDiagMass<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
		deviceData.kineticInfo.masses,
		dynKineticIndices.GetData(),
		preconditioner.GetData(), activeNodeCount);
	CUDA_CHECK(cudaPeekAtLastError());

	const auto ctx = BuildContext();
	for (auto* pContributor : m_contributors)
	{
		pContributor->AccumulateDiagonal(ctx, preconditioner.GetData());
	}

	kernel_FinalizePreconditioner<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
		isFixed.GetData(), preconditioner.GetData(), activeNodeCount);
	CUDA_CHECK(cudaPeekAtLastError());
}

double DynamicsSolveStep::ReduceDot(mcore::Vector3* a, mcore::Vector3* b, size_t count)
{
	cudaMemset(globalDotBuffer.GetData(), 0, sizeof(double));

	const auto gridCount = mcore::DivUp<DOT_BLOCK_ELEMENTS>(count);
	kernel_DotProduct<DOT_BLOCK_SIZE, DOT_EPT><<<gridCount, DOT_BLOCK_SIZE>>>(
		a, b, globalDotBuffer.GetData(), count);
	CUDA_CHECK(cudaPeekAtLastError());

	double result = 0.0;
	cudaMemcpy(&result, globalDotBuffer.GetData(), sizeof(double), cudaMemcpyDeviceToHost);
	return result;
}

void DynamicsSolveStep::CGSolve()
{
	if (activeNodeCount == 0)
		return;

	const auto& dynSysContainer = GetManager()->GetDeviceContainer<DeviceDynamicsSystemContainer>();
	const auto deviceData = dynSysContainer->GetDeviceData();

	const auto nodeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeNodeCount);
	const double tol2 = tolerance * tolerance;

	ComputePreconditioner();

	kernel_InitPCG<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
		deltaV.GetData(), residual.GetData(), z.GetData(), direction.GetData(),
		rhs.GetData(), preconditioner.GetData(), activeNodeCount);
	CUDA_CHECK(cudaPeekAtLastError());

	double rz = ReduceDot(residual.GetData(), z.GetData(), activeNodeCount);

	const auto ctx = BuildContext();

	for (IndexType iter = 0; iter < maxIterations; ++iter)
	{
		if (rz < tol2) break;

		kernel_ComputeAp_Mass<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
			deviceData.kineticInfo.masses,
			dynKineticIndices.GetData(),
			direction.GetData(), Ap.GetData(), activeNodeCount);
		CUDA_CHECK(cudaPeekAtLastError());

		for (auto* pContributor : m_contributors)
		{
			pContributor->AccumulateHessianVector(ctx, direction.GetData(), Ap.GetData());
		}

		kernel_ApplyFixedAp<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
			isFixed.GetData(), direction.GetData(), Ap.GetData(), activeNodeCount);
		CUDA_CHECK(cudaPeekAtLastError());

		const double dAp = ReduceDot(direction.GetData(), Ap.GetData(), activeNodeCount);
		if (fabs(dAp) < 1e-30) break;

		const double alpha = rz / dAp;

		kernel_CGUpdateSolution<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
			deltaV.GetData(), residual.GetData(), direction.GetData(), Ap.GetData(),
			alpha, activeNodeCount);
		CUDA_CHECK(cudaPeekAtLastError());

		kernel_ApplyPreconditioner<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
			preconditioner.GetData(), residual.GetData(), z.GetData(), activeNodeCount);
		CUDA_CHECK(cudaPeekAtLastError());

		const double rz_new = ReduceDot(residual.GetData(), z.GetData(), activeNodeCount);
		if (rz_new < tol2) break;

		const double beta = rz_new / rz;

		kernel_CGUpdateDirection<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
			z.GetData(), direction.GetData(), beta, activeNodeCount);
		CUDA_CHECK(cudaPeekAtLastError());

		rz = rz_new;
	}
}

void DynamicsSolveStep::UpdateVelocityAndPosition()
{
	const auto& dynSysContainer = GetManager()->GetDeviceContainer<DeviceDynamicsSystemContainer>();
	const auto deviceData = dynSysContainer->GetDeviceData();

	const auto nodeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeNodeCount);
	kernel_UpdateVelocityAndPosition<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
		deviceData.meshInfo.nodes,
		deviceData.kineticInfo.velocities,
		dynNodeIndices.GetData(),
		dynKineticIndices.GetData(),
		isFixed.GetData(),
		deltaV.GetData(),
		activeNodeCount);
	CUDA_CHECK(cudaPeekAtLastError());
}
