#pragma once

#include "../MCore_interface/IRenderModel.h"

#include <array>
#include <memory>
#include <functional>

namespace mcore::render
{
	class RenderModelFactory
	{
		using Func = std::function<std::unique_ptr<IRenderModel>(IRenderModelContainer*)>;
	public:
		static RenderModelFactory& Instance();

		bool Registry(RenderModelType type, Func&& func);
		RenderModelArray Build(IRenderModelContainer* pRenderModelContainer) const;

	private:
		std::array<Func, static_cast<size_t>(RenderModelType::Size)> m_funcs;
	};
}