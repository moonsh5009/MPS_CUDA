#pragma once

#include "../MCore_util/SurfaceTexture.h"
\
#include "RenderTypeDef.h"

#include <memory>

namespace mcore
{
	class IRenderingEngine;
	class IRenderPass
	{
	public:
		IRenderPass() = delete;
		IRenderPass(IRenderingEngine* pRenderingEngine)
			: m_pRenderingEngine{ pRenderingEngine }
		{}
		virtual ~IRenderPass() = default;
		IRenderPass(const IRenderPass&) = delete;
		IRenderPass(IRenderPass&&) = default;
		IRenderPass& operator=(const IRenderPass&) = delete;
		IRenderPass& operator=(IRenderPass&&) = default;

		virtual void Initialize() = 0;

		IRenderingEngine* GetRenderingEngine() const { return m_pRenderingEngine; }

	protected:
		IRenderingEngine* m_pRenderingEngine;
	};

	using RenderPassArray = std::array<std::shared_ptr<IRenderPass>, static_cast<size_t>(render::RenderPassType::Size)>;
}