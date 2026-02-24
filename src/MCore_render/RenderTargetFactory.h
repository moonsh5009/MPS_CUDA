#pragma once

#include "../MCore_interface/IRenderTarget.h"

#include <array>
#include <memory>
#include <functional>

namespace mcore::render
{
	class RenderTargetFactory
	{
		using Func = std::function<std::unique_ptr<IRenderTarget>(IRenderingEngine*)>;
	public:
		static RenderTargetFactory& Instance();

		bool Registry(RenderTargetType type, Func&& func);
		RenderTargetArray Build(IRenderingEngine* pRenderingEngine) const;

	private:
		std::array<Func, static_cast<size_t>(RenderTargetType::Size)> m_funcs;
	};
}