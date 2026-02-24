#include "stdafx.h"
#include "RenderUniformFactory.h"

using namespace mcore;
using namespace mcore::render;

RenderUniformFactory& RenderUniformFactory::Instance()
{
	static RenderUniformFactory singleton;
	return singleton;
}

bool RenderUniformFactory::Registry(RenderUniformType type, Func&& func, BuildFunc&& buildFunc)
{
	const auto id = static_cast<size_t>(type);
	assert(!m_funcs[id]);
	m_funcs[id] = std::move(func);
	m_buildFuncs[id] = std::move(buildFunc);
	return true;
}

RenderUniformArray RenderUniformFactory::Build(IScene* pScene) const
{
	RenderUniformArray result;
	for (size_t i = 0; i < m_funcs.size(); ++i)
	{
		const auto& func = m_funcs[i];
		if (!func) continue;

		result[i] = func(pScene);
	}
	return result;
}

mvk::BindGroupLayout RenderUniformFactory::BuildBindGroupLayout() const
{
	mvk::BindGroupLayoutBuilder builder;
	for (size_t i = 0; i < m_funcs.size(); ++i)
	{
		const auto& buildFunc = m_buildFuncs[i];
		if (!buildFunc) continue;
		builder = buildFunc(std::move(builder));
	}
	return std::move(builder).Build();
}
