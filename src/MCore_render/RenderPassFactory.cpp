#include "stdafx.h"
#include "RenderPassFactory.h"

using namespace mcore;
using namespace mcore::render;

RenderPassFactory& RenderPassFactory::Instance()
{
	static RenderPassFactory singleton;
	return singleton;
}

bool RenderPassFactory::Registry(RenderPassType type, Func&& func)
{
	const auto id = static_cast<size_t>(type);
	assert(!m_funcs[id]);
	m_funcs[id] = std::move(func);
	return true;
}

RenderPassArray RenderPassFactory::Build(IRenderingEngine* pRenderingEngine) const
{
	RenderPassArray result;
	for (size_t i = 0; i < m_funcs.size(); ++i)
	{
		const auto& func = m_funcs[i];
		if (!func) continue;

		result[i] = func(pRenderingEngine);
	}
	return result;
}
