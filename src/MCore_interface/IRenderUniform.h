#pragma once

#include "../MCore_util/BindGroupLayout.h"

#include "RenderTypeDef.h"

#include <memory>

namespace mcore
{
	class IScene;
	class IRenderUniform
	{
	public:
		IRenderUniform() = delete;
		IRenderUniform(IScene* pScene)
			: m_pScene{ pScene }
		{}
		virtual ~IRenderUniform() = default;
		IRenderUniform(const IRenderUniform&) = default;
		IRenderUniform(IRenderUniform&&) = default;
		IRenderUniform& operator=(const IRenderUniform&) = default;
		IRenderUniform& operator=(IRenderUniform&&) = default;

		virtual void Initialize() = 0;
		virtual bool Update(const vk::CommandBuffer& commandBuffer) = 0;

		virtual mvk::BindGroupLayoutBatchBinder&& BindBufferToLayout(mvk::BindGroupLayoutBatchBinder&& binder) const = 0;

		IScene* GetScene() const { return m_pScene; }

	protected:
		IScene* m_pScene;
	};

	using RenderUniformArray = std::array<std::unique_ptr<IRenderUniform>, static_cast<size_t>(render::RenderUniformType::Size)>;
}