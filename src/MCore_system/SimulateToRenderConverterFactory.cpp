#include "stdafx.h"
#include "SimulateToRenderConverterFactory.h"

using namespace mcore;
using namespace mcore::system;

SimulateToRenderConverterFactory& SimulateToRenderConverterFactory::Instance()
{
	static SimulateToRenderConverterFactory singleton;
	return singleton;
}

bool SimulateToRenderConverterFactory::Registry(DBTypeID id, Func&& func)
{
	if (id >= TYPE_ID_MAX)
	{
		std::cerr << "Database ID exceeds TYPE_ID_MAX\n";
		return false;
	}

	m_funcs[id] = std::move(func);
	return true;
}

SimulateToRenderConverterArray SimulateToRenderConverterFactory::Build()
{
	SimulateToRenderConverterArray result;
	result.resize(m_funcs.size());
	for (DBTypeID id = 0; id < TYPE_ID_MAX; ++id)
	{
		if (const auto& func = m_funcs[id])
		{
			result[id] = func();
		}
	}
	return result;
}
