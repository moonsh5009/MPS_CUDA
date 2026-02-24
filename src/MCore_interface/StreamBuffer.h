#pragma once

#include "../MCore_util/BufferDef.h"
#include "../MCore_util/VKDeviceBuffer.h"
#include "../MCore_util/GraphicsPipeline.h"
#include "../MCore_util/RenderContext.h"
#include "../MCore_util/StagedBuffer.h"

#include "DBDef.h"

namespace mcore
{
	template<class ATTRIBUTE, class INDIRECT>
	class StreamBuffer
	{
	public:
		StreamBuffer()
			: m_ibo{ nullptr }
			, m_position{ nullptr }
			, m_attribute{ nullptr }
			, m_indirect{ nullptr }
		{}
		~StreamBuffer() = default;
		StreamBuffer(const StreamBuffer&) = delete;
		StreamBuffer(StreamBuffer&&) = default;
		StreamBuffer& operator=(const StreamBuffer&) = delete;
		StreamBuffer& operator=(StreamBuffer&&) = default;

		virtual void Draw(const mvk::GraphicsPipeline& pipeline, const vk::CommandBuffer& commandBuffer, mvk::RenderContext* pRenderContext, uint32_t instance = 0) const = 0;

		constexpr bool IsEmpty() const { return !m_ibo || m_ibo->IsEmpty(); }

		const mcuda::VKDeviceBuffer<IndexType>& GetIBO() const { return *m_ibo; }
		const mcuda::VKDeviceBuffer<mcore::Vector3>& GetPosition() const { return *m_position; }
		const mcuda::VKDeviceBuffer<ATTRIBUTE>& GetAttribute() const { return *m_attribute; }
		const mcuda::VKDeviceBuffer<INDIRECT>& GetIndirect() const { return *m_indirect; }

		void SetIBO(const mcuda::VKDeviceBuffer<IndexType>* ibo) { m_ibo = ibo; }
		void SetPosition(const mcuda::VKDeviceBuffer<mcore::Vector3>* position) { m_position = position; }
		void SetAttribute(const mcuda::VKDeviceBuffer<ATTRIBUTE>* attribute) { m_attribute = attribute; }
		void SetIndirect(const mcuda::VKDeviceBuffer<INDIRECT>* indirect) { m_indirect = indirect; }

	protected:
		const mcuda::VKDeviceBuffer<IndexType>* m_ibo;
		const mcuda::VKDeviceBuffer<mcore::Vector3>* m_position;
		const mcuda::VKDeviceBuffer<ATTRIBUTE>* m_attribute;
		const mcuda::VKDeviceBuffer<INDIRECT>* m_indirect;
	};

	class TriangleStreamBuffer : public StreamBuffer<mvk::vbo::TriangleAttribute, mvk::DrawIndexedIndirectCommand>
	{
	public:
		TriangleStreamBuffer()
			: m_normal{ nullptr } {}

		const mcuda::VKDeviceBuffer<mcore::Vector3>& GetNormal() const { return *m_normal; }
		void SetNormal(const mcuda::VKDeviceBuffer<mcore::Vector3>* normal) { m_normal = normal; }

		void Draw(const mvk::GraphicsPipeline& pipeline, const vk::CommandBuffer& commandBuffer, mvk::RenderContext* pRenderContext, uint32_t instance) const override
		{
			if (!m_ibo || !m_position || !m_normal || !m_attribute || !m_indirect)
				return;

			const auto viewport = pRenderContext->GetViewport();
			const auto scissor = pRenderContext->GetRenderArea();

			vk::Buffer vertexBuffers[] = { *m_position, *m_normal, *m_attribute };
			vk::DeviceSize offsets[] = { 0, 0, 0 };

			std::vector<vk::DescriptorSet> sets;
			sets.reserve(pipeline.GetLayout().GetSetCount());
			sets.emplace_back(pipeline.GetLayout().GetBindGroup(0)->GetSet(0));
			assert(sets.size() == static_cast<size_t>(pipeline.GetLayout().GetSetCount()));

			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			commandBuffer.setViewportWithCount(1, &viewport);
			commandBuffer.setScissorWithCount(1, &scissor);
			commandBuffer.bindVertexBuffers(0, 3, vertexBuffers, offsets);
			commandBuffer.bindIndexBuffer(*m_ibo, 0, vk::IndexType::eUint32);
			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				pipeline.GetLayout(),
				0,
				sets,
				nullptr);
			commandBuffer.drawIndexedIndirect(*m_indirect, 0, static_cast<uint32_t>(m_indirect->GetSize()), sizeof(mvk::DrawIndexedIndirectCommand));
		}

	protected:
		const mcuda::VKDeviceBuffer<mcore::Vector3>* m_normal;
	};

	class LineStreamBuffer : public StreamBuffer<mvk::vbo::LineAttribute, mvk::DrawIndirectCommand>
	{
	public:
		LineStreamBuffer()
			: m_vertexOffset{ nullptr }
		{}

		const mcuda::VKDeviceBuffer<IndexType>& GetVertexOffset() const { return *m_vertexOffset; }
		void SetVertexOffset(const mcuda::VKDeviceBuffer<IndexType>* vertexOffset) { m_vertexOffset = vertexOffset; }

		void Draw(const mvk::GraphicsPipeline& pipeline, const vk::CommandBuffer& commandBuffer, mvk::RenderContext* pRenderContext, uint32_t instance) const override
		{
			if (!m_ibo || !m_position || !m_vertexOffset)
				return;

			const auto descriptorInstance = instance * 3 + pRenderContext->GetInFlightIndex();

			const auto viewport = pRenderContext->GetViewport();
			const auto scissor = pRenderContext->GetRenderArea();

			if (m_attribute)
			{
				pipeline.GetLayout().GetBindGroup(1)->Binder()
					.SetStorageBuffer(0, GetVertexOffset(), descriptorInstance)
					.SetStorageBuffer(1, GetIBO(), descriptorInstance)
					.SetStorageBuffer(2, GetPosition(), descriptorInstance)
					.SetStorageBuffer(3, GetAttribute(), descriptorInstance)
					.Commit();
			}
			else
			{
				pipeline.GetLayout().GetBindGroup(1)->Binder()
					.SetStorageBuffer(0, GetVertexOffset(), descriptorInstance)
					.SetStorageBuffer(1, GetIBO(), descriptorInstance)
					.SetStorageBuffer(2, GetPosition(), descriptorInstance)
					.SetNone(3, vk::DescriptorType::eStorageBuffer, descriptorInstance)
					.Commit();
			}

			std::vector<vk::DescriptorSet> sets;
			sets.reserve(pipeline.GetLayout().GetSetCount());
			sets.emplace_back(pipeline.GetLayout().GetBindGroup(0)->GetSet(0));
			sets.emplace_back(pipeline.GetLayout().GetBindGroup(1)->GetSet(descriptorInstance));
			assert(sets.size() == static_cast<size_t>(pipeline.GetLayout().GetSetCount()));

			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			commandBuffer.setViewportWithCount(1, &viewport);
			commandBuffer.setScissorWithCount(1, &scissor);
			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				pipeline.GetLayout(),
				0,
				sets,
				nullptr);
			commandBuffer.drawIndirect(*m_indirect, 0, static_cast<uint32_t>(m_indirect->GetSize()), sizeof(mvk::DrawIndirectCommand));
		}

	protected:
		const mcuda::VKDeviceBuffer<IndexType>* m_vertexOffset;
	};

	class PointStreamBuffer : public StreamBuffer<mvk::vbo::PointAttribute, mvk::DrawIndirectCommand>
	{
	public:
		PointStreamBuffer()
			: m_vertexOffset{ nullptr }
		{}

		const mcuda::VKDeviceBuffer<IndexType>& GetVertexOffset() const { return *m_vertexOffset; }
		void SetVertexOffset(const mcuda::VKDeviceBuffer<IndexType>* vertexOffset) { m_vertexOffset = vertexOffset; }

		void Draw(const mvk::GraphicsPipeline& pipeline, const vk::CommandBuffer& commandBuffer, mvk::RenderContext* pRenderContext, uint32_t instance) const override
		{
			if (!m_ibo || !m_position || !m_vertexOffset)
				return;

			const auto descriptorInstance = instance * 3 + pRenderContext->GetInFlightIndex();

			const auto viewport = pRenderContext->GetViewport();
			const auto scissor = pRenderContext->GetRenderArea();
			
			if (m_attribute)
			{
				pipeline.GetLayout().GetBindGroups()[1]->Binder()
					.SetStorageBuffer(0, GetVertexOffset(), descriptorInstance)
					.SetStorageBuffer(1, GetIBO(), descriptorInstance)
					.SetStorageBuffer(2, GetPosition(), descriptorInstance)
					.SetStorageBuffer(3, GetAttribute(), descriptorInstance)
					.Commit();
			}
			else
			{
				pipeline.GetLayout().GetBindGroup(1)->Binder()
					.SetStorageBuffer(0, GetVertexOffset(), descriptorInstance)
					.SetStorageBuffer(1, GetIBO(), descriptorInstance)
					.SetStorageBuffer(2, GetPosition(), descriptorInstance)
					.SetNone(3, vk::DescriptorType::eStorageBuffer, descriptorInstance)
					.Commit();
			}

			std::vector<vk::DescriptorSet> sets;
			sets.reserve(pipeline.GetLayout().GetSetCount());
			sets.emplace_back(pipeline.GetLayout().GetBindGroup(0)->GetSet(0));
			sets.emplace_back(pipeline.GetLayout().GetBindGroup(1)->GetSet(descriptorInstance));
			assert(sets.size() == pipeline.GetLayout().GetSetCount());

			commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			commandBuffer.setViewportWithCount(1, &viewport);
			commandBuffer.setScissorWithCount(1, &scissor);
			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				pipeline.GetLayout(),
				0,
				sets,
				nullptr);
			commandBuffer.drawIndirect(*m_indirect, 0, static_cast<uint32_t>(m_indirect->GetSize()), sizeof(mvk::DrawIndirectCommand));
		}

	protected:
		const mcuda::VKDeviceBuffer<IndexType>* m_vertexOffset;
	};
}