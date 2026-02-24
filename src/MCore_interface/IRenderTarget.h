#pragma once

#include "../MCore_util/VulkanDef.h"

#include "RenderTypeDef.h"

#include <memory>

namespace mcore
{
	class IRenderingEngine;
	class IRenderTarget
	{
	public:
		IRenderTarget() = delete;
		IRenderTarget(IRenderingEngine* pRenderingEngine)
			: m_pRenderingEngine{ pRenderingEngine }
		{}
		virtual ~IRenderTarget() = default;
		IRenderTarget(const IRenderTarget&) = delete;
		IRenderTarget(IRenderTarget&&) = default;
		IRenderTarget& operator=(const IRenderTarget&) = delete;
		IRenderTarget& operator=(IRenderTarget&&) = default;

		virtual void Initialize() = 0;
		virtual void Resize(unsigned width, unsigned height) = 0;

		IRenderingEngine* GetRenderingEngine() const { return m_pRenderingEngine; }

	protected:
		IRenderingEngine* m_pRenderingEngine;
	};

	using RenderTargetArray = std::array<std::unique_ptr<IRenderTarget>, static_cast<size_t>(render::RenderTargetType::Size)>;
}