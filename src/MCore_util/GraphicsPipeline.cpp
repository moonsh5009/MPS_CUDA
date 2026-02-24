#include "stdafx.h"
#include "GraphicsPipeline.h"

#include "VulkanCore.h"

#include <fstream>
#include <filesystem>

mvk::GraphicsPipeline::GraphicsPipeline(mvk::PipelineLayout&& pipelineLayout, vk::Pipeline pipeline)
	: m_pipeline(pipeline)
	, m_layout(std::move(pipelineLayout))
{}

mvk::GraphicsPipeline::~GraphicsPipeline()
{
	Destroy();
}

mvk::GraphicsPipeline::GraphicsPipeline(GraphicsPipeline&& other) noexcept
{
	*this = std::move(other);
}

mvk::GraphicsPipeline& mvk::GraphicsPipeline::operator=(GraphicsPipeline&& other) noexcept
{
    if (this != &other)
    {
        m_pipeline = other.m_pipeline;
        m_layout = std::move(other.m_layout);
        other.m_pipeline = nullptr;
	}
	return *this;
}

void mvk::GraphicsPipeline::Destroy()
{
    const auto pCore = VulkanCore::Instance();

    if (m_pipeline) pCore->GetDevice().destroyPipeline(m_pipeline);
    m_layout.Destroy();
    m_pipeline = nullptr;
}

vk::PipelineCache mvk::GraphicsPipelineBuilder::LoadCache(vk::Device device, const std::string& filename)
{
    if (!std::filesystem::exists(filename))
    {
        vk::PipelineCacheCreateInfo info{};
        auto cache = device.createPipelineCache(info);
        SaveCache(device, cache, filename);
        return cache;
    }

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    const auto size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);

    vk::PipelineCacheCreateInfo info{ {}, size_t(data.size()), data.data() };
    return device.createPipelineCache(info);
}

void mvk::GraphicsPipelineBuilder::SaveCache(vk::Device device, vk::PipelineCache cache, const std::string& filename)
{
    const auto data = device.getPipelineCacheData(cache);
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}

mvk::GraphicsPipelineBuilder::GraphicsPipelineBuilder(const std::string& name) :
    m_name{ PIPELINE_CACHE_DIRECTORY.data() + name + ".bin"},
    m_viewportState{},
    m_rasterizer{
        {}, VK_FALSE, VK_FALSE,
        vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eNone,
        vk::FrontFace::eClockwise,
        VK_FALSE, 0, 0, 0, 1.0f
    },
    m_multisample{ {}, vk::SampleCountFlagBits::e1 },
    m_depthStencil{},
    m_colorBlend{}
{
    /*vk::Viewport viewport{ 0, 0, 1, 1, 0, 1 };
    vk::Rect2D scissor{ { 0, 0 }, { 1, 1 } };
    m_viewportState = vk::PipelineViewportStateCreateInfo{ {}, viewport, scissor };*/
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::AddShaderModule(mvk::ShaderModule&& shaderModule)&&
{
    m_shaderModules.emplace_back(std::move(shaderModule));
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetVertexInput(const mvk::VertexInputState& vertexInput)&&
{
    m_vertexStateInput = vertexInput;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetVertexInput(mvk::VertexInputState&& vertexInput)&&
{
    m_vertexStateInput = std::move(vertexInput);
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetPrimitiveTopology(vk::PrimitiveTopology topology)&&
{
    if (!m_inputAssembly) m_inputAssembly = vk::PipelineInputAssemblyStateCreateInfo{ {}, {}, vk::False };
    m_inputAssembly->topology = topology;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetPrimitiveRestart(bool primitiveRestartEnable)&&
{
    if (!m_inputAssembly) m_inputAssembly = vk::PipelineInputAssemblyStateCreateInfo{ {}, {}, vk::False };
    m_inputAssembly->primitiveRestartEnable = primitiveRestartEnable;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetCullMode(vk::CullModeFlags cullMode)&&
{
    m_rasterizer.cullMode = cullMode;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetInputAssembly(vk::PipelineInputAssemblyStateCreateInfo assembly)&&
{
	m_inputAssembly = assembly;
	return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetViewportState(vk::PipelineViewportStateCreateInfo viewportState)&&
{
	m_viewportState = viewportState;
	return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetRasterizer(vk::PipelineRasterizationStateCreateInfo rasterizer)&&
{
	m_rasterizer = rasterizer;
	return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetMultisample(vk::PipelineMultisampleStateCreateInfo multisample)&&
{
	m_multisample = multisample;
	return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetDepthStencil(vk::PipelineDepthStencilStateCreateInfo depthStencil)&&
{
	m_depthStencil = depthStencil;
	return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetDepthState(bool testEnable, bool writeEnable, vk::CompareOp compareOp, const std::optional<std::pair<float, float>>& depthBounds)&&
{
    m_depthStencil.depthTestEnable = testEnable;
    m_depthStencil.depthWriteEnable = writeEnable;
    m_depthStencil.depthCompareOp = compareOp;
    if (depthBounds)
    {
        m_depthStencil.depthBoundsTestEnable = vk::True;
        m_depthStencil.minDepthBounds = depthBounds->first;
        m_depthStencil.maxDepthBounds = depthBounds->second;
    }
    else
    {
        m_depthStencil.depthBoundsTestEnable = vk::False;
    }
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetColorBlend(const ColorBlendState& blend)&&
{
    m_colorBlend = blend;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetColorBlend(ColorBlendState&& blend)&&
{
    m_colorBlend = std::move(blend);
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetPipelineLayout(mvk::PipelineLayout&& layout)&&
{
    m_layout = std::move(layout);
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::AetDynamicState(vk::DynamicState dynamicState)&&
{
    m_dynamicStates.emplace_back(dynamicState);
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetDynamicStates(std::vector<vk::DynamicState>&& dynamicStates)&&
{
    m_dynamicStates = std::move(dynamicStates);
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::AddColorAttachment(vk::Format format)&&
{
    m_colorAttachmentFormats.push_back(format);
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetColorAttachments(const std::vector<vk::Format>& formats)&&
{
    m_colorAttachmentFormats = formats;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetDepthAttachment(vk::Format format)&&
{
    m_depthAttachmentFormat = format;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetStencilAttachment(vk::Format format)&&
{
    m_stencilAttachmentFormat = format;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetDepthNStencilAttachment(vk::Format format)&&
{
    m_depthAttachmentFormat = format;
    m_stencilAttachmentFormat = format;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetRenderingInfo(const vk::PipelineRenderingCreateInfo& renderingInfo)&&
{
    m_renderingInfo = renderingInfo;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetRenderPass(vk::RenderPass renderPass, uint32_t subpass)&&
{
    m_renderPass = renderPass;
    m_subpass = subpass;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetBasePipeline(vk::Pipeline basePipeline)&&
{
    m_basePipeline = basePipeline;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetBasePipelineIndex(int32_t index)&&
{
    m_basePipelineIndex = index;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::AllowDerivatives()&&
{
    m_flags |= vk::PipelineCreateFlagBits::eAllowDerivatives;
    return std::move(*this);
}

mvk::GraphicsPipelineBuilder&& mvk::GraphicsPipelineBuilder::SetAsDerivative()&&
{
    m_flags |= vk::PipelineCreateFlagBits::eDerivative;
    return std::move(*this);
}

mvk::GraphicsPipeline mvk::GraphicsPipelineBuilder::Build()&&
{
    const auto pCore = VulkanCore::Instance();

    if (!m_layout || !m_layout->Get())
    {
        m_layout = mvk::PipelineLayoutBuilder{}
            .Build();
    }

    ValidateConfiguration();

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    shaderStages.reserve(m_shaderModules.size());
    for (const auto& shaderModule : m_shaderModules)
    {
        shaderStages.emplace_back(shaderModule.Get());
    }

    if (IsUsingDynamicRendering())
    {
        BuildRenderingInfo();
    }
    vk::PipelineDynamicStateCreateInfo dynamicState{
        {},
        m_dynamicStates
    };

    vk::GraphicsPipelineCreateInfo info{
        m_flags,
        shaderStages,
        m_vertexStateInput.has_value() ? &m_vertexStateInput->Get() : nullptr,
        m_inputAssembly.has_value() ? &m_inputAssembly.value() : nullptr,
        m_tessellation.has_value() ? &m_tessellation.value() : nullptr,
        &m_viewportState,
        &m_rasterizer,
        &m_multisample,
        &m_depthStencil,
        m_colorBlend.has_value() ? &m_colorBlend->Get() : nullptr,
        m_dynamicStates.empty() ? nullptr : &dynamicState,
        m_layout->Get(),
        m_renderPass,
        m_subpass,
        m_basePipeline,
        m_basePipelineIndex,
        m_renderingInfo.has_value() ? &m_renderingInfo.value() : nullptr,
    };

    const auto cache = GraphicsPipelineBuilder::LoadCache(pCore->GetDevice(), m_name);
    vk::Pipeline pipeline = pCore->GetDevice().createGraphicsPipeline(cache, info).value;
    pCore->GetDevice().destroyPipelineCache(cache);
    return GraphicsPipeline{ std::move(*m_layout), pipeline };
}

void mvk::GraphicsPipelineBuilder::BuildRenderingInfo()
{
    if (!m_renderingInfo.has_value())
    {
        m_renderingInfo = vk::PipelineRenderingCreateInfo{
            {},
            m_colorAttachmentFormats,
            m_depthAttachmentFormat,
            m_stencilAttachmentFormat
        };
    }
}

bool mvk::GraphicsPipelineBuilder::IsUsingDynamicRendering() const
{
    return !m_colorAttachmentFormats.empty() ||
        m_depthAttachmentFormat != vk::Format::eUndefined ||
        m_stencilAttachmentFormat != vk::Format::eUndefined ||
        m_renderingInfo.has_value();
}

void mvk::GraphicsPipelineBuilder::ValidateConfiguration() const
{
    if (m_shaderModules.empty())
    {
        throw std::runtime_error("At least one shader module is required");
    }

    if (!m_layout.has_value())
    {
        throw std::runtime_error("Pipeline layout is required");
    }

    bool hasDynamicRendering = IsUsingDynamicRendering();
    bool hasRenderPass = (m_renderPass != VK_NULL_HANDLE);

    if (hasDynamicRendering && hasRenderPass)
    {
        throw std::runtime_error("Cannot use both dynamic rendering and render pass");
    }

    if (!hasDynamicRendering && !hasRenderPass)
    {
        throw std::runtime_error("Either dynamic rendering info or render pass is required");
    }

    if (m_basePipeline != VK_NULL_HANDLE && !(m_flags & vk::PipelineCreateFlagBits::eDerivative))
    {
        throw std::runtime_error("Base pipeline specified but derivative flag not set");
    }
}