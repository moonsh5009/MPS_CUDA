#pragma once

#include "../MCore_interface/IScene.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ Scene : public IScene
	{
	public:
		Scene(IRenderCore* pRenderCore);

		void Initialize(std::shared_ptr<mvk::RenderContext>&& pRenderContext) override;
		void OnResize(unsigned width, unsigned height) override;

		void Draw() override;
		void Invalidate() override;
		void SetZoomFit() override;

		void UpdateViewport() override;
		bool UpdateAABB(const vk::CommandBuffer& commandBuffer) override;
		bool UpdateUniform(const vk::CommandBuffer& commandBuffer) override;

		void SetCameraMode(CameraDirectionType dirType, bool is3D) const override;

		IRenderModelContainer* GetModelContainer() const;

		const glm::uvec4& GetViewport() const override { return m_viewport; }
		const glm::vec4& GetBackgroundColor() const override { return m_backgroundColor; }
		const AABBf& GetAABB() const override { return m_aabb; }

		vk::Format GetSwapchainFormat() const override;
		vk::Format GetColorFormat() const override;
		vk::Format GetDepthFormat() const override;
		vk::SampleCountFlagBits GetMultiSampleCount() const override;
		vk::PipelineMultisampleStateCreateInfo GetMultiSampleState() const override;

	private:
		void InitUniform();
		void InitAABB();

		bool Resize();

		glm::uvec4 m_viewport;
		glm::vec4 m_backgroundColor;
		AABBf m_aabb;
	};
}

#include "HeaderPost.h"