#include "stdafx.h"
#include "SimulateStepFactory.h"

#include <algorithm>

using namespace mcore;
using namespace mcore::simulate;

SimulateStepFactory& SimulateStepFactory::Instance()
{
	static SimulateStepFactory singleton;
	return singleton;
}

bool SimulateStepFactory::Registry(uint32_t order, Func&& func)
{
	m_entries.push_back({ order, std::move(func) });
	return true;
}

SimulateStepGroupArray SimulateStepFactory::Build(ISimulateManager* pManager)
{
	std::sort(m_entries.begin(), m_entries.end(),
		[](const Entry& a, const Entry& b) { return a.order < b.order; });

	SimulateStepGroupArray result;

	uint32_t currentOrder = UINT32_MAX;
	for (auto& entry : m_entries)
	{
		if (entry.order != currentOrder)
		{
			currentOrder = entry.order;
			result.emplace_back();
		}
		auto pStep = entry.func(pManager);
		pStep->Initialize();
		result.back().push_back(std::move(pStep));
	}

	for (auto& group : result)
		for (auto& pStep : group)
			pStep->PostInitialize();

	return result;
}
