#pragma once

#include "../MCore_simulate/SimulateStep.h"
#include "../MCore_interface/IDynamicsContributor.h"

#include "DeviceClothDynamicsContainer.h"
#include "DeviceDynamicsSystemContainer.h"
#include "DeviceMeshContainer.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ ClothDynamicsContributor
	: public mcore::simulate::SimulateStep
	, public mcore::IDynamicsContributor
{
public:
	ClothDynamicsContributor(mcore::ISimulateManager* pManager);

	void Initialize() override;
	void Execute(cudaStream_t stream, REAL dt) override;
	void OnUpdatedContainer() override;

	void AccumulateGradient(const mcore::DynamicsContext& ctx, mcore::Vector3* rhs) override;
	void AccumulateHessianVector(const mcore::DynamicsContext& ctx, const mcore::Vector3* p, mcore::Vector3* Ap) override;
	void AccumulateDiagonal(const mcore::DynamicsContext& ctx, mcore::Vector3* diag) override;
	void AccumulateDamping(const mcore::DynamicsContext& ctx, mcore::Vector3* rhs) override;

	void UpdateActiveIndices();

	mcuda::DeviceBuffer<IndexType> activeEdgeNodeIndices;
	mcuda::DeviceBuffer<double> activeRestLengths;
	mcuda::DeviceBuffer<BendEdge> activeBendEdges;

	size_t activeNodeCount = 0;
	size_t activeEdgeCount = 0;
	size_t activeBendEdgeCount = 0;
	double activeStretchStiffness = 1000.0;
	double activeBendStiffness = 0.01;
	double activeDampingCoeff = 0.01;
};

#include "HeaderPost.h"
