#pragma once

#include "IRenderTarget.h"
#include "IRenderPass.h"

namespace mcore
{
	class IScene;
	class IRenderModelContainer;
	class IRenderingEngine
	{
	public:
		IRenderingEngine(IScene* pScene) 
			: m_pScene{ pScene }
		{}
		virtual ~IRenderingEngine() = default;
		IRenderingEngine(const IRenderingEngine&) = delete;
		IRenderingEngine(IRenderingEngine&&) = default;
		IRenderingEngine& operator=(const IRenderingEngine&) = delete;
		IRenderingEngine& operator=(IRenderingEngine&&) = default;

		virtual void Initialize() = 0;
		virtual void OnResize(unsigned width, unsigned height) = 0;

		virtual void Draw() = 0;
		virtual void Invalidate() = 0;

		IScene* GetScene() const { return m_pScene; }

	protected:
		IScene* m_pScene;

	public:
		template<class RENDER_TARGET>
		RENDER_TARGET* GetTarget() const { return static_cast<RENDER_TARGET*>(m_targets[RENDER_TARGET::id].get()); }
		const RenderTargetArray& GetTargets() const { return m_targets; }

		template<class RENDER_PASS>
		RENDER_PASS* GetRenderPass() const { return static_cast<RENDER_PASS*>(m_renderPasses[RENDER_PASS::id].get()); }
		const RenderPassArray& GetRenderPasses() const { return m_renderPasses; }

	protected:
		RenderTargetArray m_targets;
		RenderPassArray m_renderPasses;
	};
}