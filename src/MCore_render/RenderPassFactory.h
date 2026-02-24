#pragma once

#include "../MCore_interface/IRenderPass.h"

#include <array>
#include <memory>
#include <functional>

namespace mcore::render
{
	class RenderPassFactory
	{
		using Func = std::function<std::shared_ptr<IRenderPass>(IRenderingEngine*)>;
	public:
		static RenderPassFactory& Instance();

		bool Registry(RenderPassType type, Func&& func);
		RenderPassArray Build(IRenderingEngine* pRenderingEngine) const;

	private:
		std::array<Func, static_cast<size_t>(RenderPassType::Size)> m_funcs;
	};
}