#pragma once

#include "VulkanDef.h"

#include "HeaderPre.h"

namespace mvk
{
    class __MY_EXT_CLASS__ VertexInputState
    {
    public:
        VertexInputState() = delete;
        VertexInputState(
            vk::PipelineVertexInputStateCreateFlags flags,
            std::vector<vk::VertexInputBindingDescription>&& bindingDescriptions,
            std::vector<vk::VertexInputAttributeDescription>&& attributeDescriptions);
        VertexInputState(const VertexInputState&);
        VertexInputState(VertexInputState&&) noexcept;
        VertexInputState& operator=(const VertexInputState&);
        VertexInputState& operator=(VertexInputState&&) noexcept;

        const vk::PipelineVertexInputStateCreateInfo& Get() const { return m_info; }
        operator vk::PipelineVertexInputStateCreateInfo() const { return m_info; }

    private:
        vk::PipelineVertexInputStateCreateInfo m_info;
        std::vector<vk::VertexInputBindingDescription> m_bindingDescriptions;
        std::vector<vk::VertexInputAttributeDescription> m_attributeDescriptions;
    };

    class __MY_EXT_CLASS__ VertexInputStateBuilder
    {
    public:
        VertexInputStateBuilder() = default;
        VertexInputStateBuilder(const VertexInputStateBuilder&) = delete;
        VertexInputStateBuilder(VertexInputStateBuilder&&) noexcept = default;
        VertexInputStateBuilder& operator=(const VertexInputStateBuilder&) = delete;
        VertexInputStateBuilder& operator=(VertexInputStateBuilder&&) noexcept = default;

        VertexInputStateBuilder&& SetFlags(vk::PipelineVertexInputStateCreateFlags createFlags)&&;
        VertexInputStateBuilder&& AddBinding(uint32_t binding, uint32_t stride, vk::VertexInputRate inputRate = vk::VertexInputRate::eVertex)&&;
        VertexInputStateBuilder&& AddBindings(const std::vector<vk::VertexInputBindingDescription>& bindings)&&;
        VertexInputStateBuilder&& AddAttribute(uint32_t binding, uint32_t location, vk::Format format, uint32_t offset)&&;
        VertexInputStateBuilder&& AddAttribute(const vk::VertexInputAttributeDescription& attrDesc)&&;
        VertexInputStateBuilder&& AddAttributes(const std::vector<vk::VertexInputAttributeDescription>& attributes)&&;
        VertexInputStateBuilder&& AddFloatAttribute(uint32_t binding, uint32_t location, uint32_t offset)&&;
        VertexInputStateBuilder&& AddFloat2Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&;
        VertexInputStateBuilder&& AddFloat3Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&;
        VertexInputStateBuilder&& AddFloat4Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&;
        VertexInputStateBuilder&& AddDoubleAttribute(uint32_t binding, uint32_t location, uint32_t offset)&&;
        VertexInputStateBuilder&& AddDouble2Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&;
        VertexInputStateBuilder&& AddDouble3Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&;
        VertexInputStateBuilder&& AddDouble4Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&;
        VertexInputStateBuilder&& AddColorAttribute(uint32_t binding, uint32_t location, uint32_t offset)&&;

        mvk::VertexInputState Build()&&;

    private:
        vk::PipelineVertexInputStateCreateFlags m_flags{};
        std::vector<vk::VertexInputBindingDescription> m_bindingDescriptions;
        std::vector<vk::VertexInputAttributeDescription> m_attributeDescriptions;
    };
}

#include "HeaderPost.h"