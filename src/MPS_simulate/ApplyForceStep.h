#pragma once

#include "../MCore_simulate/SimulateStep.h"

#include "DeviceForceDynamicsContainer.h"
#include "DeviceMeshContainer.h"
#include "DeviceKineticContainer.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ ApplyForceStep : public mcore::simulate::SimulateStep
{
public:
	ApplyForceStep(mcore::ISimulateManager* pManager);

	void Initialize() override;
	void Execute(cudaStream_t stream, REAL dt) override;
	void OnUpdatedContainer() override;

	void UpdateActiveNodeIndices();

	void InitForce();
	void ApplyGravity();
	void IntegrateForce();

	mcuda::DeviceBuffer<IndexType> activeNodeIndices;
	mcuda::DeviceBuffer<IndexType> activeKineticIndices;
	mcuda::DeviceBuffer<IndexType> activeFixeds;
};

#include "HeaderPost.h"
