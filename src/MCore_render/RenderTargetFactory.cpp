#include "stdafx.h"
#include "RenderTargetFactory.h"

using namespace mcore;
using namespace mcore::render;

RenderTargetFactory& RenderTargetFactory::Instance()
{
	static RenderTargetFactory singleton;
	return singleton;
}

bool RenderTargetFactory::Registry(RenderTargetType type, Func&& func)
{
	const auto id = static_cast<size_t>(type);
	assert(!m_funcs[id]);
	m_funcs[id] = std::move(func);
	return true;
}

RenderTargetArray RenderTargetFactory::Build(IRenderingEngine* pRenderingEngine) const
{
	RenderTargetArray result;
	for (size_t i = 0; i < m_funcs.size(); ++i)
	{
		const auto& func = m_funcs[i];
		if (!func) continue;

		result[i] = func(pRenderingEngine);
	}
	return result;
}
