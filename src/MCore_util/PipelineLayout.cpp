#include "stdafx.h"
#include "PipelineLayout.h"

#include "VulkanCore.h"

mvk::PipelineLayout::PipelineLayout(std::vector<mvk::BindGroupLayout>&& layouts, vk::PipelineLayout pipelineLayout)
    : m_layouts(std::move(layouts))
    , m_pipelineLayout(pipelineLayout)
{}

mvk::PipelineLayout::~PipelineLayout()
{
    Destroy();
}

mvk::PipelineLayout::PipelineLayout(PipelineLayout&& other)
{
	*this = std::move(other);
}

mvk::PipelineLayout& mvk::PipelineLayout::operator=(PipelineLayout&& other)
{
    if (this != &other)
    {
        m_layouts = std::move(other.m_layouts);
        m_pipelineLayout = other.m_pipelineLayout;
		other.m_pipelineLayout = nullptr;
	}
	return *this;
}

void mvk::PipelineLayout::Destroy()
{
    const auto pCore = VulkanCore::Instance();

    if (m_pipelineLayout)
    {
        pCore->GetDevice().destroyPipelineLayout(m_pipelineLayout);
        m_pipelineLayout = VK_NULL_HANDLE;
    }

    m_layouts.clear();
}

std::string mvk::PipelineLayout::GetLayoutDescription() const
{
    std::string desc = "Pipeline Layout:\n";

    /*for (size_t setIdx = 0; setIdx < m_bindingInfos.size(); ++setIdx)
    {
        desc += std::format("  Set {}: {} instances\n", setIdx,
            setIdx < m_extraSets.size() ? m_extraSets[setIdx].size() : 0);

        for (const auto& binding : m_bindingInfos[setIdx])
        {
            desc += std::format("    Binding {}: {} ({}) - Stages: {}\n",
                binding.binding,
                static_cast<uint32_t>(binding.type),
                binding.name.empty() ? "unnamed" : binding.name,
                static_cast<uint32_t>(binding.stageFlags));
        }
    }*/

    return desc;
}

mvk::PipelineLayoutBuilder&& mvk::PipelineLayoutBuilder::AddBindGroupLayout(uint32_t set, const mvk::BindGroupLayout& layout)&&
{
    if (m_bindGroupLayouts.size() <= set)
    {
        m_bindGroupLayouts.resize(set + 1);
    }

    m_bindGroupLayouts[set] = layout;
    return std::move(*this);
}

mvk::PipelineLayoutBuilder&& mvk::PipelineLayoutBuilder::AddConstantRange(vk::ShaderStageFlags stageFlags, uint32_t offset, uint32_t size)&&
{
    m_pushConstants.emplace_back(stageFlags, offset, size);
    return std::move(*this);
}

mvk::PipelineLayout mvk::PipelineLayoutBuilder::Build()&&
{
    const auto pCore = VulkanCore::Instance();

    if (m_bindGroupLayouts.empty())
    {
        vk::PipelineLayout pipelineLayout = pCore->GetDevice().createPipelineLayout(
            vk::PipelineLayoutCreateInfo{ {}, {}, m_pushConstants });
        return { {}, pipelineLayout };
    }

    std::vector<vk::DescriptorSetLayout> layouts;
    layouts.reserve(m_bindGroupLayouts.size());
    std::ranges::transform(m_bindGroupLayouts, std::back_inserter(layouts), [](const auto& bindGroupLayout) -> vk::DescriptorSetLayout
    {
        return bindGroupLayout ? bindGroupLayout->Get() : VK_NULL_HANDLE;
    });

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ {}, layouts, m_pushConstants };
    const auto pipelineLayout = pCore->GetDevice().createPipelineLayout(pipelineLayoutInfo);
    return { std::move(m_bindGroupLayouts), pipelineLayout };
}

std::string mvk::PipelineLayoutBuilder::GetDescription() const
{
    std::string desc = "Pipeline Layout Builder:\n";

    /*if (!m_bindings.empty())
    {
        desc += std::format("  Current Set ({}): {} bindings\n", m_setBindings.size(), m_bindings.size());
        for (const auto& binding : m_bindings)
        {
            desc += std::format("    Binding {}: {} - {}\n",
                binding.binding,
                static_cast<uint32_t>(binding.type),
                binding.name.empty() ? "unnamed" : binding.name);
        }
    }

    for (size_t setIdx = 0; setIdx < m_setBindings.size(); ++setIdx)
    {
        desc += std::format("  Set {}: {} bindings\n", setIdx, m_setBindings[setIdx].size());
        for (const auto& binding : m_setBindings[setIdx])
        {
            desc += std::format("    Binding {}: {} - {}\n",
                binding.binding,
                static_cast<uint32_t>(binding.type),
                binding.name.empty() ? "unnamed" : binding.name);
        }
    }

    if (!m_pushConstants.empty())
    {
        desc += "  Push Constants:\n";
        for (const auto& pc : m_pushConstants)
        {
            desc += std::format("    Offset: {}, Size: {}, Stages: {}\n",
                pc.offset, pc.size, static_cast<uint32_t>(pc.stageFlags));
        }
    }*/

    return desc;
}