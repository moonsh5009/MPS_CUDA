#pragma once

#include "../MCore_interface/IRenderCore.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ RenderCore : public IRenderCore
	{
	public:
		void Initialize() override;
		void AddScene(HWND window) override;
		void Run() override;
		void Invalidate() override;
		void SetZoomFitAllScenes() override;

		vk::Format GetColorFormat() const override;
		vk::Format GetDepthFormat() const override;
		vk::SampleCountFlagBits GetMultiSampleCount() const override;
		vk::PipelineMultisampleStateCreateInfo GetMultiSampleState() const override;

		const mvk::BindGroupLayout& GetUniformBindGrouplayout() const override;
		mvk::BindGroupLayout GetPointStreamBindGrouplayout() const override;
		mvk::BindGroupLayout GetLineStreamBindGrouplayout() const override;

	private:
		mvk::BindGroupLayout m_uniformBindGroupLayout;
	};
}

#include "HeaderPost.h"