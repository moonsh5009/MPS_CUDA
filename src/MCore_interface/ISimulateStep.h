#pragma once

#include "DBDef.h"

#include <memory>
#include <vector>

#ifndef __DRIVER_TYPES_H__
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

namespace mcore
{
	class ISimulateManager;

	class ISimulateStep
	{
	public:
		ISimulateStep(ISimulateManager* pManager)
			: m_pManager{ pManager }
		{}
		virtual ~ISimulateStep() = default;
		ISimulateStep(const ISimulateStep&) = delete;
		ISimulateStep(ISimulateStep&&) = default;
		ISimulateStep& operator=(const ISimulateStep&) = delete;
		ISimulateStep& operator=(ISimulateStep&&) = default;

		virtual void Initialize() = 0;
		virtual void PostInitialize() {}
		virtual void Execute(cudaStream_t stream, REAL dt) = 0;
		virtual void OnUpdatedContainer() {}

		ISimulateManager* GetManager() const { return m_pManager; }

	protected:
		ISimulateManager* m_pManager;
	};

	using SimulateStepGroup = std::vector<std::unique_ptr<ISimulateStep>>;
	using SimulateStepGroupArray = std::vector<SimulateStepGroup>;
}
