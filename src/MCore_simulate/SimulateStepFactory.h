#pragma once

#include "../MCore_interface/ISimulateStep.h"

#include <functional>

#include "HeaderPre.h"

namespace mcore::simulate
{
	class __MY_EXT_CLASS__ SimulateStepFactory
	{
	public:
		using Func = std::function<std::unique_ptr<ISimulateStep>(ISimulateManager*)>;

		static SimulateStepFactory& Instance();

		bool Registry(uint32_t order, Func&& func);
		SimulateStepGroupArray Build(ISimulateManager* pManager);

	private:
		struct Entry
		{
			uint32_t order;
			Func func;
		};
		std::vector<Entry> m_entries;
	};
}

#include "HeaderPost.h"
