#pragma once

#include "../MCore_interface/ISimulateManager.h"

#include <cuda_runtime.h>

#include "HeaderPre.h"

namespace mcore::simulate
{
	class __MY_EXT_CLASS__ SimulateManager : public ISimulateManager
	{
	public:
		~SimulateManager() override;

		void Initialize() override;

		void Simulate() override;

		void SetDeltaTime(REAL dt) { m_dt = dt; }
		REAL GetDeltaTime() const { return m_dt; }

	private:
		void EnsureStreams(size_t count);

		std::vector<cudaStream_t> m_streams;
		REAL m_dt = 1.0 / 60.0;
	};
}

#include "HeaderPost.h"