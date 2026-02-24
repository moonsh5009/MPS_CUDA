#pragma once

#include "../MCore_util/AABB.h"
#include "../MCore_util/Signal.h"
#include "../MCore_util/RenderContext.h"

#include "RenderTypeDef.h"

#include "IRenderUniform.h"
#include "IRenderTarget.h"

#include "IRenderingEngine.h"
#include "IUserInputHandler.h"

namespace mcore
{
	class IRenderCore;
	class IScene
	{
	public:
		IScene() = delete;
		IScene(IRenderCore* pRenderCore)
			: m_pRenderCore{ pRenderCore }
		{}
		virtual ~IScene() = default;
		IScene(const IScene&) = delete;
		IScene(IScene&&) = default;
		IScene& operator=(const IScene&) = delete;
		IScene& operator=(IScene&&) = default;

		virtual void Initialize(std::shared_ptr<mvk::RenderContext>&& pRenderContext) = 0;
		virtual void OnResize(unsigned width, unsigned height) = 0;

		virtual void Draw() = 0;
		virtual void Invalidate() = 0;
		virtual void SetZoomFit() = 0;

		virtual void UpdateViewport() = 0;
		virtual bool UpdateAABB(const vk::CommandBuffer& commandBuffer) = 0;
		virtual bool UpdateUniform(const vk::CommandBuffer& commandBuffer) = 0;

		virtual void SetCameraMode(render::CameraDirectionType dirType, bool is3D) const = 0;

		virtual const glm::uvec4& GetViewport() const = 0;
		virtual const glm::vec4& GetBackgroundColor() const = 0;
		virtual const AABBf& GetAABB() const = 0;

		virtual vk::Format GetSwapchainFormat() const = 0;
		virtual vk::Format GetColorFormat() const = 0;
		virtual vk::Format GetDepthFormat() const = 0;
		virtual vk::SampleCountFlagBits GetMultiSampleCount() const = 0;
		virtual vk::PipelineMultisampleStateCreateInfo GetMultiSampleState() const = 0;

		mcore::Signal<void(const AABBf&)> onUpdateAABB = mcore::MakeSignal<void(const AABBf&)>();

		IRenderCore* GetRenderCore() const { return m_pRenderCore; }

	protected:
		IRenderCore* m_pRenderCore;

	public:
		const std::shared_ptr<mvk::RenderContext>& GetRenderContext() const { return m_pRenderContext; }
		IRenderingEngine* GetRenderingEngine() const { return m_pRenderingEngine.get(); }
		IUserInputHandler* GetUserInputHandler() const { return m_pUserInputHandler.get(); }

		template<class RENDER_UNIFORM>
		RENDER_UNIFORM* GetUniform() const { return static_cast<RENDER_UNIFORM*>(m_uniforms[RENDER_UNIFORM::id].get()); }
		const RenderUniformArray& GetUniforms() const { return m_uniforms; }

	protected:
		std::shared_ptr<mvk::RenderContext> m_pRenderContext;
		std::unique_ptr<IRenderingEngine> m_pRenderingEngine;
		std::unique_ptr<IUserInputHandler> m_pUserInputHandler;
		RenderUniformArray m_uniforms;
	};
}