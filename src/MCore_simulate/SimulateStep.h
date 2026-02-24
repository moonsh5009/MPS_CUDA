#pragma once

#include "../MCore_interface/ISimulateStep.h"

#include "SimulateStepFactory.h"

namespace mcore::simulate
{
	class SimulateStep : public ISimulateStep
	{
	public:
		using ISimulateStep::ISimulateStep;

		void Initialize() override {}
		void Execute(cudaStream_t stream, REAL dt) override {}
	};
}

#define REGISTRY_SIMULATE_STEP(STEP_CLASS, ORDER) \
	namespace { \
		const bool registered_##STEP_CLASS = \
			mcore::simulate::SimulateStepFactory::Instance().Registry(ORDER, \
			[](mcore::ISimulateManager* pManager) -> std::unique_ptr<mcore::ISimulateStep> \
			{ return std::make_unique<STEP_CLASS>(pManager); }); \
	}
