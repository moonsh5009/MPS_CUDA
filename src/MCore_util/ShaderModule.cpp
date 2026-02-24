#include "stdafx.h"
#include "ShaderModule.h"

#include "VulkanCore.h"

#include <fstream>

mvk::ShaderModule::ShaderModule(const std::string& path)
{
    const auto fullPath = SHADER_DIRECTORY.data() + path + ".spv";
    std::ifstream file(fullPath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Failed to open shader file: " + fullPath);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<uint32_t> code(fileSize / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(code.data()), fileSize);

    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = fileSize;
    createInfo.pCode = code.data();

    m_module = VulkanCore::Instance()->GetDevice().createShaderModule(createInfo);
    m_shaderStageCreateInfo.setModule(m_module);

    m_entryPoint = "main";
    m_shaderStageCreateInfo.setPName(m_entryPoint.data());
}

mvk::ShaderModule::~ShaderModule()
{
    if (m_module)
    {
        VulkanCore::Instance()->GetDevice().destroyShaderModule(m_module);
    }
}

mvk::ShaderModule::ShaderModule(ShaderModule&& other) noexcept
{
	*this = std::move(other);
}

mvk::ShaderModule& mvk::ShaderModule::operator=(ShaderModule&& other) noexcept
{
    if (this != &other)
    {
        m_module = std::move(other.m_module);
        m_entryPoint = std::move(other.m_entryPoint);
        m_constantEntries = std::move(other.m_constantEntries);
        m_constantData = std::move(other.m_constantData);
        m_specializationInfo = other.m_specializationInfo;
        m_shaderStageCreateInfo = other.m_shaderStageCreateInfo;
        m_specializationInfo.setMapEntries(m_constantEntries);
        m_specializationInfo.setData<char>(m_constantData);

        other.m_module = nullptr;
    }
	return *this;
}

void mvk::ShaderModule::SetEntryPoint(const std::string& entryPoint)
{
    m_entryPoint = entryPoint;
}

void mvk::ShaderModule::SetStage(vk::ShaderStageFlagBits stage)
{
    m_shaderStageCreateInfo.setStage(stage);
}

void mvk::ShaderModule::SetConstantEntries(std::vector<vk::SpecializationMapEntry>&& constantEntries)
{
    m_constantEntries = std::move(constantEntries);
    m_specializationInfo.setMapEntries(m_constantEntries);
}

void mvk::ShaderModule::SetConstantData(std::vector<char>&& data)
{
    m_constantData = std::move(data);
    m_specializationInfo.setData<char>(m_constantData);
}

vk::PipelineShaderStageCreateInfo mvk::ShaderModule::Get() const
{
    vk::PipelineShaderStageCreateInfo shaderStageCreateInfo = m_shaderStageCreateInfo;
    shaderStageCreateInfo.setPName(m_entryPoint.c_str());

    if (!m_constantData.empty())
        shaderStageCreateInfo.setPSpecializationInfo(&m_specializationInfo);

    return shaderStageCreateInfo;
}
