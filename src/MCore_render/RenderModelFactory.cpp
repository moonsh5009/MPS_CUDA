#include "stdafx.h"
#include "RenderModelFactory.h"

using namespace mcore;
using namespace mcore::render;

RenderModelFactory& RenderModelFactory::Instance()
{
	static RenderModelFactory singleton;
	return singleton;
}

bool RenderModelFactory::Registry(RenderModelType type, Func&& func)
{
	const auto id = static_cast<size_t>(type);
	assert(!m_funcs[id]);
	m_funcs[id] = std::move(func);
	return true;
}

RenderModelArray RenderModelFactory::Build(IRenderModelContainer* pRenderModelContainer) const
{
	RenderModelArray result;
	for (size_t i = 0; i < m_funcs.size(); ++i)
	{
		const auto& func = m_funcs[i];
		if (!func) continue;

		result[i] = func(pRenderModelContainer);
	}
	return result;
}
