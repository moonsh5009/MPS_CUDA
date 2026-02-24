#pragma once

#include "../MCore_simulate/SimulateStep.h"
#include "../MCore_interface/IDynamicsContributor.h"

#include "DeviceDynamicsSystemContainer.h"
#include "DeviceMeshContainer.h"
#include "DeviceKineticContainer.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ DynamicsSolveStep : public mcore::simulate::SimulateStep
{
public:
	DynamicsSolveStep(mcore::ISimulateManager* pManager);

	void Initialize() override;
	void PostInitialize() override;
	void Execute(cudaStream_t stream, REAL dt) override;
	void OnUpdatedContainer() override;

	void RegisterContributor(mcore::IDynamicsContributor* pContributor);

	void UpdateActiveIndices();
	void ApplyGravity();
	void ComputeElasticGradient();
	void ComputePreconditioner();
	void CGSolve();
	void UpdateVelocityAndPosition();
	double ReduceDot(mcore::Vector3* a, mcore::Vector3* b, size_t count);

	mcore::DynamicsContext BuildContext() const;

	mcuda::DeviceBuffer<IndexType> dynNodeIndices;
	mcuda::DeviceBuffer<IndexType> dynKineticIndices;
	mcuda::DeviceBuffer<IndexType> isFixed;

	mcuda::DeviceBuffer<mcore::Vector3> deltaV;
	mcuda::DeviceBuffer<mcore::Vector3> rhs;
	mcuda::DeviceBuffer<mcore::Vector3> residual;
	mcuda::DeviceBuffer<mcore::Vector3> direction;
	mcuda::DeviceBuffer<mcore::Vector3> Ap;
	mcuda::DeviceBuffer<mcore::Vector3> preconditioner;
	mcuda::DeviceBuffer<mcore::Vector3> z;

	mcuda::DeviceBuffer<double> globalDotBuffer;

	size_t activeNodeCount = 0;
	IndexType maxIterations = 30;
	double tolerance = 1e-8;

	std::vector<mcore::IDynamicsContributor*> m_contributors;
};

#include "HeaderPost.h"
