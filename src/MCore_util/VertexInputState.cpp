#include "stdafx.h"
#include "VertexInputState.h"

mvk::VertexInputState::VertexInputState(
    vk::PipelineVertexInputStateCreateFlags flags,
    std::vector<vk::VertexInputBindingDescription>&& bindingDescriptions,
    std::vector<vk::VertexInputAttributeDescription>&& attributeDescriptions)
    : m_bindingDescriptions{ std::move(bindingDescriptions) }
    , m_attributeDescriptions{std::move(attributeDescriptions) }
{
    m_info.flags = flags;
    m_info.vertexBindingDescriptionCount = static_cast<uint32_t>(m_bindingDescriptions.size());
    m_info.pVertexBindingDescriptions = m_bindingDescriptions.data();
    m_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(m_attributeDescriptions.size());
    m_info.pVertexAttributeDescriptions = m_attributeDescriptions.data();
}

mvk::VertexInputState::VertexInputState(const VertexInputState& other)
{
    *this = other;
}

mvk::VertexInputState::VertexInputState(VertexInputState&& other) noexcept
{
    *this = std::move(other);
}

mvk::VertexInputState& mvk::VertexInputState::operator=(const VertexInputState& other)
{
    if (this != &other)
    {
        m_bindingDescriptions = other.m_bindingDescriptions;
        m_attributeDescriptions = other.m_attributeDescriptions;

        m_info.flags = other.m_info.flags;
        m_info.vertexBindingDescriptionCount = static_cast<uint32_t>(m_bindingDescriptions.size());
        m_info.pVertexBindingDescriptions = m_bindingDescriptions.data();
        m_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(m_attributeDescriptions.size());
        m_info.pVertexAttributeDescriptions = m_attributeDescriptions.data();
    }
    return *this;
}

mvk::VertexInputState& mvk::VertexInputState::operator=(VertexInputState&& other) noexcept
{
    if (this != &other)
    {
        m_bindingDescriptions = std::move(other.m_bindingDescriptions);
        m_attributeDescriptions = std::move(other.m_attributeDescriptions);

        other.m_info.vertexBindingDescriptionCount = 0;
        other.m_info.pVertexBindingDescriptions = nullptr;
        other.m_info.vertexAttributeDescriptionCount = 0;
        other.m_info.pVertexAttributeDescriptions = nullptr;

        m_info.flags = other.m_info.flags;
        m_info.vertexBindingDescriptionCount = static_cast<uint32_t>(m_bindingDescriptions.size());
        m_info.pVertexBindingDescriptions = m_bindingDescriptions.data();
        m_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(m_attributeDescriptions.size());
        m_info.pVertexAttributeDescriptions = m_attributeDescriptions.data();
    }
    return *this;
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::SetFlags(vk::PipelineVertexInputStateCreateFlags createFlags)&&
{
    m_flags = createFlags;
    return std::move(*this);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddBinding(uint32_t binding, uint32_t stride, vk::VertexInputRate inputRate)&&
{
    vk::VertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = binding;
    bindingDesc.stride = stride;
    bindingDesc.inputRate = inputRate;
    m_bindingDescriptions.emplace_back(bindingDesc);
    return std::move(*this);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddBindings(const std::vector<vk::VertexInputBindingDescription>& bindings)&&
{
    m_bindingDescriptions.insert(m_bindingDescriptions.end(), bindings.begin(), bindings.end());
    return std::move(*this);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddAttribute(uint32_t binding, uint32_t location, vk::Format format, uint32_t offset)&&
{
    vk::VertexInputAttributeDescription attrDesc{};
    attrDesc.location = location;
    attrDesc.binding = binding;
    attrDesc.format = format;
    attrDesc.offset = offset;
    m_attributeDescriptions.emplace_back(attrDesc);
    return std::move(*this);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddAttribute(const vk::VertexInputAttributeDescription& attrDesc)&&
{
    m_attributeDescriptions.push_back(attrDesc);
    return std::move(*this);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddAttributes(const std::vector<vk::VertexInputAttributeDescription>& attributes)&&
{
    m_attributeDescriptions.insert(m_attributeDescriptions.end(), attributes.begin(), attributes.end());
    return std::move(*this);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddFloatAttribute(uint32_t binding, uint32_t location, uint32_t offset)&&
{
    return std::move(*this).AddAttribute(binding, location, vk::Format::eR32Sfloat, offset);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddFloat2Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&
{
    return std::move(*this).AddAttribute(binding, location, vk::Format::eR32G32Sfloat, offset);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddFloat3Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&
{
    return std::move(*this).AddAttribute(binding, location, vk::Format::eR32G32B32Sfloat, offset);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddFloat4Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&
{
    return std::move(*this).AddAttribute(binding, location, vk::Format::eR32G32B32A32Sfloat, offset);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddDoubleAttribute(uint32_t binding, uint32_t location, uint32_t offset)&&
{
	return std::move(*this).AddAttribute(binding, location, vk::Format::eR64Sfloat, offset);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddDouble2Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&
{
	return std::move(*this).AddAttribute(binding, location, vk::Format::eR64G64Sfloat, offset);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddDouble3Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&
{
	return std::move(*this).AddAttribute(binding, location, vk::Format::eR64G64B64Sfloat, offset);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddDouble4Attribute(uint32_t binding, uint32_t location, uint32_t offset)&&
{
	return std::move(*this).AddAttribute(binding, location, vk::Format::eR64G64B64A64Sfloat, offset);
}

mvk::VertexInputStateBuilder&& mvk::VertexInputStateBuilder::AddColorAttribute(uint32_t binding, uint32_t location, uint32_t offset)&&
{
	return std::move(*this).AddAttribute(binding, location, vk::Format::eR8G8B8A8Unorm, offset);
}

mvk::VertexInputState mvk::VertexInputStateBuilder::Build()&&
{
    return VertexInputState{ m_flags, std::move(m_bindingDescriptions), std::move(m_attributeDescriptions) };
}