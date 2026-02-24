#include "stdafx.h"
#include "DeviceContainerFactory.h"

using namespace mcore;
using namespace mcore::simulate;

DeviceContainerFactory& DeviceContainerFactory::Instance()
{
	static DeviceContainerFactory singleton;
	return singleton;
}

bool DeviceContainerFactory::Registry(DBTypeID id, Func&& func)
{
	if (id >= TYPE_ID_MAX)
	{
		std::cerr << "Database ID exceeds TYPE_ID_MAX\n";
		return false;
	}

	m_funcs[id] = std::move(func);
	return true;
}

DeviceContainerArray DeviceContainerFactory::Build(ISimulateManager* pSimulateManager)
{
	DeviceContainerArray result;
	result.resize(m_funcs.size());
	for (DBTypeID id = 0; id < TYPE_ID_MAX; ++id)
	{
		if (const auto& func = m_funcs[id])
		{
			result[id] = func(pSimulateManager);
		}
	}
	return result;
}
