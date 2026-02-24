#pragma once

#include "../MCore_simulate/SimulateStep.h"

#include "DeviceForceDynamicsContainer.h"

#include "HeaderPre.h"

class ApplyForceStep;

class __MY_EXT_CLASS__ IntegratePositionStep : public mcore::simulate::SimulateStep
{
public:
	IntegratePositionStep(mcore::ISimulateManager* pManager);

	void Initialize() override;
	void PostInitialize() override;
	void Execute(cudaStream_t stream, REAL dt) override;

	void IntegrateVelocity();

private:
	ApplyForceStep* m_pApplyForceStep = nullptr;
};

#include "HeaderPost.h"
