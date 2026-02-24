#pragma once

#include "BindGroupLayout.h"

#include "HeaderPre.h"

namespace mvk
{
    class __MY_EXT_CLASS__ PipelineLayout
    {
    public:
        PipelineLayout() = default;
        PipelineLayout(std::vector<mvk::BindGroupLayout>&& layouts,vk::PipelineLayout pipelineLayout);
        PipelineLayout(const PipelineLayout&) = delete;
        PipelineLayout(PipelineLayout&& other);
        PipelineLayout& operator=(const PipelineLayout&) = delete;
        PipelineLayout& operator=(PipelineLayout&& other);
        ~PipelineLayout();

        void Destroy();

        std::string GetLayoutDescription() const;

        bool IsValid() const { return m_pipelineLayout; }
        uint32_t GetSetCount() const { return static_cast<uint32_t>(m_layouts.size()); }

        operator vk::PipelineLayout() const { return m_pipelineLayout; }
        vk::PipelineLayout Get() const { return m_pipelineLayout; }
        const mvk::BindGroupLayout& GetBindGroup(uint32_t setIndex) const { return m_layouts[setIndex]; }
        const std::vector<mvk::BindGroupLayout>& GetBindGroups() const { return m_layouts; }

    private:
        friend class PipelineLayoutBatchUpdater;

        std::vector<mvk::BindGroupLayout> m_layouts;
        vk::PipelineLayout m_pipelineLayout;
    };

    class __MY_EXT_CLASS__ PipelineLayoutBuilder final
    {
    public:
        PipelineLayoutBuilder() = default;
        ~PipelineLayoutBuilder() = default;
        PipelineLayoutBuilder(const PipelineLayoutBuilder&) = delete;
        PipelineLayoutBuilder(PipelineLayoutBuilder&&) = default;
        PipelineLayoutBuilder& operator=(const PipelineLayoutBuilder&) = delete;
        PipelineLayoutBuilder& operator=(PipelineLayoutBuilder&&) = default;

        PipelineLayoutBuilder&& AddBindGroupLayout(uint32_t set, const mvk::BindGroupLayout& layout)&&;
        PipelineLayoutBuilder&& AddConstantRange(vk::ShaderStageFlags stageFlags, uint32_t offset, uint32_t size)&&;

        PipelineLayout Build()&&;

        std::string GetDescription() const;

    private:
        std::vector<mvk::BindGroupLayout> m_bindGroupLayouts;
        std::vector<vk::PushConstantRange> m_pushConstants;

        std::vector<vk::DescriptorPoolSize> CalculatePoolSizes(uint32_t numInstances) const;
    };
}

#include "HeaderPost.h"