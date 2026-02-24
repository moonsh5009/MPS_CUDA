#pragma once

#include "../MCore_util/PersistentMappedBuffer.h"
#include "../MCore_util/StagedBuffer.h"

#include "RenderUniformFactory.h"

namespace mcore::render
{
	template<typename HOST_DATA>
	class RenderUniformBase : public IRenderUniform
	{
	public:
		RenderUniformBase(IScene* pScene)
			: IRenderUniform{ pScene }
			, m_bDirty{ true }
			, m_ubo{ vk::BufferUsageFlagBits::eUniformBuffer }
		{}

		mvk::PersistentMappedBuffer<HOST_DATA>& GetUBO() { return m_ubo; }
		const mvk::PersistentMappedBuffer<HOST_DATA>& GetUBO() const { return m_ubo; }

		void CopyFrom(const vk::CommandBuffer& commandBuffer, const mvk::PersistentMappedBuffer<HOST_DATA>& src)
		{
			m_ubo.CopyFromDevice(commandBuffer, src, { 0, 0, sizeof(HOST_DATA) });
		}
		void CopyFrom(const vk::CommandBuffer& commandBuffer, const mvk::StagedBuffer<HOST_DATA>& src)
		{
			assert(src.GetSize() == 1);
			m_ubo.CopyFromDevice(commandBuffer, src, { 0, 0, sizeof(HOST_DATA) });
		}

	protected:
		void SetDirty() { m_bDirty = true; }
		void ClearDirty() { m_bDirty = false; }
		bool IsDirty() const { return m_bDirty; }

    private:
        bool m_bDirty;
		mvk::PersistentMappedBuffer<HOST_DATA> m_ubo;
    };
}

#define DECLARE_RENDER_UNIFORM(RENDER_UNIFORM) \
	public: \
		static size_t id; \
        static mvk::BindGroupLayoutBuilder&& BuildBindGroupLayout(mvk::BindGroupLayoutBuilder&& builder); \
        mvk::BindGroupLayoutBatchBinder&& BindBufferToLayout(mvk::BindGroupLayoutBatchBinder&& binder) const override; \
	private:

#define IMPLEMENT_RENDER_UNIFORM(RENDER_UNIFORM, TYPE, SHADER_STAGE) \
	size_t RENDER_UNIFORM::id = static_cast<size_t>(TYPE); \
	namespace \
	{ \
		const auto registry_##RENDER_UNIFORM = mcore::render::RenderUniformFactory::Instance().Registry(TYPE, [](auto pScene) \
		{ \
			return std::make_unique<RENDER_UNIFORM>(pScene); \
		}, [](mvk::BindGroupLayoutBuilder&& builder) -> mvk::BindGroupLayoutBuilder&& \
		{ \
			return RENDER_UNIFORM::BuildBindGroupLayout(std::move(builder)); \
		}); \
	} \
	mvk::BindGroupLayoutBuilder&& RENDER_UNIFORM::BuildBindGroupLayout(mvk::BindGroupLayoutBuilder&& builder) \
	{ \
		return std::move(builder) \
			.AddUniformBuffer( \
				static_cast<uint32_t>(TYPE), \
				SHADER_STAGE, \
				#RENDER_UNIFORM \
			); \
	} \
	mvk::BindGroupLayoutBatchBinder&& RENDER_UNIFORM::BindBufferToLayout(mvk::BindGroupLayoutBatchBinder&& binder) const \
	{ \
		return std::move(binder) \
			.SetUniformBuffer( \
				static_cast<uint32_t>(id), \
				GetUBO() \
			); \
	}