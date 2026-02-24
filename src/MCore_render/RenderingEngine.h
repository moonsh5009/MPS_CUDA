#pragma once

#include "../MCore_interface/IRenderingEngine.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ RenderingEngine : public IRenderingEngine
	{
	public:
		RenderingEngine(IScene* pScene);

		void Initialize() override;
		void OnResize(unsigned width, unsigned height) override;

		void Draw() override;
		void Invalidate() override;

	private:
		void InitTarget();
		void InitRenderPass();

		bool Resize();
		void Update();

		bool m_bDraw = true;
		std::optional<vk::Extent2D> m_windowResize;
	};
}

#include "HeaderPost.h"