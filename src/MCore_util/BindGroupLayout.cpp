#include "stdafx.h"
#include "BindGroupLayout.h"

#include "VulkanCore.h"

mvk::BindGroupLayoutBatchBinder::BindGroupLayoutBatchBinder(const BindGroupLayoutImpl* layout)
    : m_layout{ layout }
{}

mvk::BindGroupLayoutBatchBinder&& mvk::BindGroupLayoutBatchBinder::Reserve(size_t count)&&
{
    m_bufferInfos.reserve(count);
    m_imageInfos.reserve(count);
    return std::move(*this);
}

mvk::BindGroupLayoutBatchBinder&& mvk::BindGroupLayoutBatchBinder::SetNone(uint32_t binding,
    vk::DescriptorType type, uint32_t instance)&&
{
    m_bufferInfos.emplace_back(vk::DescriptorBufferInfo{ VK_NULL_HANDLE, 0, VK_WHOLE_SIZE }, type, binding, instance);
    return std::move(*this);
}

mvk::BindGroupLayoutBatchBinder&& mvk::BindGroupLayoutBatchBinder::SetBuffer(uint32_t binding,
    vk::Buffer buffer, vk::DeviceSize size, vk::DescriptorType type, uint32_t instance)&&
{
    m_bufferInfos.emplace_back(vk::DescriptorBufferInfo{ buffer, 0, size }, type, binding, instance);
    return std::move(*this);
}

mvk::BindGroupLayoutBatchBinder&& mvk::BindGroupLayoutBatchBinder::SetImage(uint32_t binding,
    vk::Sampler sampler, vk::ImageView imageView, vk::ImageLayout layout, uint32_t instance)&&
{
    m_imageInfos.emplace_back(vk::DescriptorImageInfo{ sampler, imageView, layout },
        vk::DescriptorType::eCombinedImageSampler, binding, instance);
    return std::move(*this);
}

mvk::BindGroupLayoutBatchBinder&& mvk::BindGroupLayoutBatchBinder::SetStorageImage(uint32_t binding,
    vk::ImageView imageView, vk::ImageLayout layout, uint32_t instance)&&
{
    m_imageInfos.emplace_back(vk::DescriptorImageInfo{ VK_NULL_HANDLE, imageView, layout },
        vk::DescriptorType::eStorageImage, binding, instance);
    return std::move(*this);
}

void mvk::BindGroupLayoutBatchBinder::Commit()&&
{
    std::vector<vk::WriteDescriptorSet> pendingWrites;
    pendingWrites.reserve(m_bufferInfos.size() + m_imageInfos.size());
    for (const auto& [info, type, binding, instance] : m_bufferInfos)
    {
        vk::WriteDescriptorSet write{};
        write.dstSet = m_layout->GetSet(instance);
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorType = type;
        write.descriptorCount = 1;
        write.pBufferInfo = &info;
        pendingWrites.emplace_back(write);
    }
    for (const auto& [info, type, binding, instance] : m_imageInfos)
    {
        vk::WriteDescriptorSet write{};
        write.dstSet = m_layout->GetSet(instance);
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorType = type;
        write.descriptorCount = 1;
        write.pImageInfo = &info;
        pendingWrites.emplace_back(write);
    }
    if (!pendingWrites.empty())
    {
        VulkanCore::Instance()->GetDevice().updateDescriptorSets(pendingWrites, nullptr);
    }
}

mvk::BindGroupLayoutImpl::BindGroupLayoutImpl(
    vk::DescriptorPool pool,
    vk::DescriptorSetLayout layout,
    std::vector<vk::DescriptorSet>&& sets,
    std::vector<vk::DescriptorSetLayoutBinding>&& bindings,
    std::vector<std::string>&& bindingLabels)
    : m_pool{ pool }
    , m_layout{ layout }
    , m_sets{ std::move(sets) }
    , m_bindings{ std::move(bindings) }
    , m_bindingLabels{ std::move(bindingLabels) }
{
}

mvk::BindGroupLayoutImpl::~BindGroupLayoutImpl()
{
    Destroy();
}

void mvk::BindGroupLayoutImpl::Destroy()
{
	const auto pCore = VulkanCore::Instance();
    if (m_layout)
    {
        pCore->GetDevice().destroyDescriptorSetLayout(m_layout);
        m_layout = VK_NULL_HANDLE;
    }

    if (m_pool)
    {
        pCore->GetDevice().destroyDescriptorPool(m_pool);
        m_pool = VK_NULL_HANDLE;
    }
}


std::string mvk::BindGroupLayoutImpl::GetDescription() const
{
    std::string desc = "BindGroup Layout:\n";
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


mvk::BindGroupLayoutBuilder&& mvk::BindGroupLayoutBuilder::Reserve(uint32_t size)&&
{
    m_bindings.reserve(size);
    m_bindingLabels.reserve(size);
    return std::move(*this);
}

mvk::BindGroupLayoutBuilder&& mvk::BindGroupLayoutBuilder::AddBinding(uint32_t binding, vk::DescriptorType type,
    vk::ShaderStageFlags stageFlags, uint32_t count, const std::string& label)&&
{
    m_bindings.emplace_back(binding, type, count, stageFlags);
    m_bindingLabels.emplace_back(label);
    return std::move(*this);
}

mvk::BindGroupLayoutBuilder&& mvk::BindGroupLayoutBuilder::AddUniformBuffer(uint32_t binding, vk::ShaderStageFlags stageFlags,
    const std::string& label)&&
{
    return std::move(*this).AddBinding(binding, vk::DescriptorType::eUniformBuffer, stageFlags, 1, label);
}

mvk::BindGroupLayoutBuilder&& mvk::BindGroupLayoutBuilder::AddStorageBuffer(uint32_t binding, vk::ShaderStageFlags stageFlags,
    const std::string& label)&&
{
    return std::move(*this).AddBinding(binding, vk::DescriptorType::eStorageBuffer, stageFlags, 1, label);
}

mvk::BindGroupLayoutBuilder&& mvk::BindGroupLayoutBuilder::AddCombinedImageSampler(uint32_t binding, vk::ShaderStageFlags stageFlags,
    uint32_t count, const std::string& label)&&
{
    return std::move(*this).AddBinding(binding, vk::DescriptorType::eCombinedImageSampler, stageFlags, count, label);
}

mvk::BindGroupLayoutBuilder&& mvk::BindGroupLayoutBuilder::AddStorageImage(uint32_t binding, vk::ShaderStageFlags stageFlags,
    const std::string& label)&&
{
    return std::move(*this).AddBinding(binding, vk::DescriptorType::eStorageImage, stageFlags, 1, label);
}

mvk::BindGroupLayout mvk::BindGroupLayoutBuilder::Build(uint32_t numInstances)&&
{
    if (!m_bindings.empty())
    {
        ValidateCurrentSet();
    }

    const auto pCore = VulkanCore::Instance();

    const auto poolSizes = CalculatePoolSizes(numInstances);
    const auto pool = pCore->GetDevice().createDescriptorPool(
        {
            m_poolFlags,
            numInstances,
            poolSizes
        });

    vk::DescriptorSetLayoutCreateInfo layoutInfo{ {}, m_bindings };
    const auto layout = pCore->GetDevice().createDescriptorSetLayout(layoutInfo);

    std::vector<vk::DescriptorSetLayout> repeatedLayouts(numInstances, layout);
    vk::DescriptorSetAllocateInfo allocInfo{ pool, repeatedLayouts };
    auto sets = pCore->GetDevice().allocateDescriptorSets(allocInfo);

    return std::make_shared<BindGroupLayoutImpl>(
        pool,
        layout,
        std::move(sets),
        std::move(m_bindings),
        std::move(m_bindingLabels));
}

std::string mvk::BindGroupLayoutBuilder::GetDescription() const
{
    std::string desc = "Pipeline Layout Builder:\n";

   /* if (!m_bindings.empty())
    {
        for (size_t i = 0; i < m_bindings.size(); ++i)
        {
            const auto& binding = m_bindings[i];
            const auto& label = m_bindingLabels[i];
            desc += std::format("    Binding {}: {} - {}\n",
                binding.binding,
                static_cast<uint32_t>(binding.descriptorType),
                label.empty() ? "unlabeld" : label);
        }
    }*/

    return desc;
}

void mvk::BindGroupLayoutBuilder::ValidateCurrentSet() const
{
    if (m_bindings.empty())
    {
        throw std::runtime_error("Cannot create empty descriptor set");
    }

    std::unordered_set<uint32_t> bindingNumbers;
    for (const auto& binding : m_bindings)
    {
        if (bindingNumbers.contains(binding.binding))
        {
            throw std::runtime_error(std::format("Duplicate binding number {} in set", binding.binding));
        }
        bindingNumbers.insert(binding.binding);
    }
}

std::vector<vk::DescriptorPoolSize> mvk::BindGroupLayoutBuilder::CalculatePoolSizes(uint32_t numInstances) const
{
    std::unordered_map<vk::DescriptorType, uint32_t> typeCounts;
    for (const auto& binding : m_bindings)
    {
        const auto num = numInstances * binding.descriptorCount;
        const auto emp = typeCounts.emplace(binding.descriptorType, num);
        if (!emp.second) emp.first->second += num;
    }

    std::vector<vk::DescriptorPoolSize> poolSizes;
    poolSizes.reserve(typeCounts.size());
    for (const auto& [type, count] : typeCounts)
    {
        poolSizes.emplace_back(type, count);
    }
    return poolSizes;
}