#include "stdafx.h"
#include "ClothDynamicsSimulator.cuh"

#include "ClothDynamicsContributor.h"

#include <thrust/sequence.h>
#include <thrust/device_ptr.h>

namespace
{
	constexpr size_t BLOCK_SIZE = 1024;
	constexpr size_t ELEMENTS_PER_THREAD = 8;
	constexpr size_t BLOCK_ELEMENTS = BLOCK_SIZE * ELEMENTS_PER_THREAD;

	constexpr size_t BEND_BLOCK_SIZE = 128;
	constexpr size_t BEND_EPT = 8;
	constexpr size_t BEND_BLOCK_ELEMENTS = BEND_BLOCK_SIZE * BEND_EPT;
}

REGISTRY_SIMULATE_STEP(ClothDynamicsContributor, 200)

ClothDynamicsContributor::ClothDynamicsContributor(mcore::ISimulateManager* pManager)
	: mcore::simulate::SimulateStep{ pManager }
{}

void ClothDynamicsContributor::Initialize()
{}

void ClothDynamicsContributor::Execute(cudaStream_t stream, REAL dt)
{}

void ClothDynamicsContributor::OnUpdatedContainer()
{
	UpdateActiveIndices();
}

void ClothDynamicsContributor::UpdateActiveIndices()
{
	const auto& clothContainer = GetManager()->GetDeviceContainer<DeviceClothDynamicsContainer>();
	const auto& dynSysContainer = GetManager()->GetDeviceContainer<DeviceDynamicsSystemContainer>();
	const auto& meshContainer = GetManager()->GetDeviceContainer<DeviceMeshContainer>();
	if (clothContainer->IsEmpty() || dynSysContainer->IsEmpty() || meshContainer->IsEmpty())
		return;

	activeNodeCount = clothContainer->activeNodeCount;
	activeEdgeCount = clothContainer->activeEdgeCount;
	activeBendEdgeCount = clothContainer->activeBendEdgeCount;

	if (activeNodeCount == 0)
		return;

	activeEdgeNodeIndices.SetSize(activeEdgeCount * 2);
	activeRestLengths.SetSize(activeEdgeCount);
	activeBendEdges.SetSize(activeBendEdgeCount);

	size_t edgeOffset = 0;
	size_t bendOffset = 0;

	for (size_t i = 0; i < clothContainer->dynamicsSystemIndices.GetSize(); ++i)
	{
		const auto dynSysIndex = clothContainer->hostDynamicsSystemIndices[static_cast<IndexType>(i)];

		if (dynSysIndex >= dynSysContainer->entryNodeOffsets.size())
			continue;

		const auto nodeOffset = static_cast<IndexType>(dynSysContainer->entryNodeOffsets[dynSysIndex]);
		const auto meshIndex = dynSysContainer->GetHostMeshIndices()[dynSysIndex];

		const auto& edgeRange = meshContainer->edgeIndices.GetRange(meshIndex);
		if (edgeRange && edgeRange->size > 0)
		{
			const auto edgeVertexCount = edgeRange->size;
			const auto edgeCountLocal = edgeVertexCount / 2;

			std::vector<IndexType> hostEdges(edgeVertexCount);
			cudaMemcpy(hostEdges.data(),
				meshContainer->edgeIndices.GetBuffer().GetData() + edgeRange->offset,
				edgeVertexCount * sizeof(IndexType),
				cudaMemcpyDeviceToHost);

			std::vector<IndexType> localEdgeNodes(edgeVertexCount);
			for (size_t e = 0; e < edgeVertexCount; ++e)
			{
				localEdgeNodes[e] = nodeOffset + hostEdges[e];
			}

			cudaMemcpy(
				activeEdgeNodeIndices.GetData() + edgeOffset * 2,
				localEdgeNodes.data(),
				edgeVertexCount * sizeof(IndexType),
				cudaMemcpyHostToDevice);

			const auto& restRange = clothContainer->restLengths.GetRange(static_cast<IndexType>(i));
			if (restRange && restRange->size == edgeCountLocal)
			{
				cudaMemcpy(
					activeRestLengths.GetData() + edgeOffset,
					clothContainer->restLengths.GetBuffer().GetData() + restRange->offset,
					edgeCountLocal * sizeof(double),
					cudaMemcpyDeviceToDevice);
			}

			edgeOffset += edgeCountLocal;
		}

		const auto& bendRange = clothContainer->bendEdges.GetRange(static_cast<IndexType>(i));
		if (bendRange && bendRange->size > 0)
		{
			const auto bendCount = bendRange->size;
			std::vector<BendEdge> hostBendEdges(bendCount);
			cudaMemcpy(hostBendEdges.data(),
				clothContainer->bendEdges.GetBuffer().GetData() + bendRange->offset,
				bendCount * sizeof(BendEdge),
				cudaMemcpyDeviceToHost);

			for (auto& be : hostBendEdges)
			{
				be.v0 = nodeOffset + be.v0;
				be.v1 = nodeOffset + be.v1;
				be.v2 = nodeOffset + be.v2;
				be.v3 = nodeOffset + be.v3;
			}

			cudaMemcpy(
				activeBendEdges.GetData() + bendOffset,
				hostBendEdges.data(),
				bendCount * sizeof(BendEdge),
				cudaMemcpyHostToDevice);

			bendOffset += bendCount;
		}

		if (i == 0)
		{
			activeStretchStiffness = clothContainer->hostStretchStiffnessArray[static_cast<IndexType>(i)];
			activeBendStiffness = clothContainer->hostBendStiffnessArray[static_cast<IndexType>(i)];
			activeDampingCoeff = clothContainer->hostDampingCoeffArray[static_cast<IndexType>(i)];
		}
	}
}

void ClothDynamicsContributor::AccumulateGradient(const mcore::DynamicsContext& ctx, mcore::Vector3* rhsOut)
{
	if (activeEdgeCount > 0)
	{
		const auto edgeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeEdgeCount);
		kernel_ComputeStretchGradient<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<edgeGridCount, BLOCK_SIZE>>>(
			ctx.meshNodes,
			ctx.dynNodeIndices,
			activeEdgeNodeIndices.GetData(),
			activeRestLengths.GetData(),
			activeStretchStiffness,
			rhsOut,
			activeEdgeCount);
		CUDA_CHECK(cudaPeekAtLastError());
	}

	if (activeBendEdgeCount > 0)
	{
		const auto bendGridCount = mcore::DivUp<BEND_BLOCK_ELEMENTS>(activeBendEdgeCount);
		kernel_ComputeBendGradient<BEND_BLOCK_SIZE, BEND_EPT><<<bendGridCount, BEND_BLOCK_SIZE>>>(
			ctx.meshNodes,
			ctx.dynNodeIndices,
			activeBendEdges.GetData(),
			activeBendStiffness,
			rhsOut,
			activeBendEdgeCount);
		CUDA_CHECK(cudaPeekAtLastError());
	}
}

void ClothDynamicsContributor::AccumulateHessianVector(const mcore::DynamicsContext& ctx, const mcore::Vector3* p, mcore::Vector3* ApOut)
{
	if (activeEdgeCount > 0)
	{
		const auto edgeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeEdgeCount);
		kernel_ComputeAp_StretchHv<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<edgeGridCount, BLOCK_SIZE>>>(
			ctx.meshNodes,
			ctx.dynNodeIndices,
			activeEdgeNodeIndices.GetData(),
			activeRestLengths.GetData(),
			activeStretchStiffness,
			p,
			ApOut,
			activeEdgeCount);
		CUDA_CHECK(cudaPeekAtLastError());
	}

	if (activeBendEdgeCount > 0)
	{
		const auto bendGridCount = mcore::DivUp<BEND_BLOCK_ELEMENTS>(activeBendEdgeCount);
		kernel_ComputeAp_BendHv<BEND_BLOCK_SIZE, BEND_EPT><<<bendGridCount, BEND_BLOCK_SIZE>>>(
			ctx.meshNodes,
			ctx.dynNodeIndices,
			activeBendEdges.GetData(),
			activeBendStiffness,
			p,
			ApOut,
			activeBendEdgeCount);
		CUDA_CHECK(cudaPeekAtLastError());
	}
}

void ClothDynamicsContributor::AccumulateDiagonal(const mcore::DynamicsContext& ctx, mcore::Vector3* diag)
{
	if (activeEdgeCount > 0)
	{
		const auto edgeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(activeEdgeCount);
		kernel_AccumDiag_Stretch<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<edgeGridCount, BLOCK_SIZE>>>(
			ctx.meshNodes,
			ctx.dynNodeIndices,
			activeEdgeNodeIndices.GetData(),
			activeRestLengths.GetData(),
			activeStretchStiffness,
			diag,
			activeEdgeCount);
		CUDA_CHECK(cudaPeekAtLastError());
	}
}

void ClothDynamicsContributor::AccumulateDamping(const mcore::DynamicsContext& ctx, mcore::Vector3* rhsOut)
{
	if (activeNodeCount == 0 || activeDampingCoeff <= 0.0)
		return;

	const auto nodeGridCount = mcore::DivUp<BLOCK_ELEMENTS>(ctx.totalNodeCount);
	kernel_AccumDamping<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<nodeGridCount, BLOCK_SIZE>>>(
		ctx.velocities,
		ctx.dynKineticIndices,
		activeDampingCoeff,
		rhsOut,
		ctx.totalNodeCount);
	CUDA_CHECK(cudaPeekAtLastError());
}
