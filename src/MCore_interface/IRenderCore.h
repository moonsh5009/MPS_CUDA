#pragma once

#include "IRenderModelContainer.h"
#include "IScene.h"

namespace mcore
{
	class IRenderCore
	{
	public:
		IRenderCore() = default;
		virtual ~IRenderCore() = default;
		IRenderCore(const IRenderCore&) = delete;
		IRenderCore(IRenderCore&&) noexcept = delete;
		IRenderCore& operator=(const IRenderCore&) = delete;
		IRenderCore& operator=(IRenderCore&&) noexcept = delete;

		virtual void Initialize() = 0;
		virtual void AddScene(HWND window) = 0;
		virtual void Run() = 0;
		virtual void Invalidate() = 0;
		virtual void SetZoomFitAllScenes() = 0;

		virtual vk::Format GetColorFormat() const = 0;
		virtual vk::Format GetDepthFormat() const = 0;
		virtual vk::SampleCountFlagBits GetMultiSampleCount() const = 0;
		virtual vk::PipelineMultisampleStateCreateInfo GetMultiSampleState() const = 0;

		virtual const mvk::BindGroupLayout& GetUniformBindGrouplayout() const = 0;
		virtual mvk::BindGroupLayout GetPointStreamBindGrouplayout() const = 0;
		virtual mvk::BindGroupLayout GetLineStreamBindGrouplayout() const = 0;

		void RemoveScene(HWND window)
		{
			const auto itr = m_scenes.find(window);
			if (itr == m_scenes.end())
				return;
			m_scenes.erase(itr);
		}

		IScene* GetScene(HWND window) const
		{
			if (const auto itr = m_scenes.find(window); itr != m_scenes.end())
			{
				return itr->second.get();
			}
			return nullptr;
		}

		const std::unordered_map<HWND, std::unique_ptr<IScene>>& GetScenes() const
		{
			return m_scenes;
		}

		IRenderModelContainer* GetModelContainer() const { return m_pModelContainer.get(); }

	protected:
		std::unique_ptr<IRenderModelContainer> m_pModelContainer;
		std::unordered_map<HWND, std::unique_ptr<IScene>> m_scenes;
	};
}