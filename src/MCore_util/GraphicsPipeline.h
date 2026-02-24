#pragma once

#include "ShaderModule.h"
#include "VertexInputState.h"
#include "ColorBlendState.h"
#include "PipelineLayout.h"

#include "HeaderPre.h"

namespace mvk
{
    class __MY_EXT_CLASS__ GraphicsPipeline
    {
    public:
        GraphicsPipeline() = default;
        GraphicsPipeline(mvk::PipelineLayout&& pipelineLayout, vk::Pipeline pipeline);
        ~GraphicsPipeline();
        GraphicsPipeline(const GraphicsPipeline&) = delete;
        GraphicsPipeline(GraphicsPipeline&&) noexcept;
        GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
        GraphicsPipeline& operator=(GraphicsPipeline&&) noexcept;

		operator bool() const { return m_pipeline; }

        void Destroy();

        vk::Pipeline Get() const { return m_pipeline; }
        operator vk::Pipeline() const { return m_pipeline; }

        constexpr const mvk::PipelineLayout& GetLayout() const { return m_layout; }

    private:
        vk::Pipeline m_pipeline;
        mvk::PipelineLayout m_layout;
    };

    class __MY_EXT_CLASS__ GraphicsPipelineBuilder
    {
    public:
        static vk::PipelineCache LoadCache(vk::Device device, const std::string& filename);
        static void SaveCache(vk::Device device, vk::PipelineCache cache, const std::string& filename);

        GraphicsPipelineBuilder() = delete;
        GraphicsPipelineBuilder(const std::string& name);
        GraphicsPipelineBuilder(const GraphicsPipelineBuilder&) = delete;
        GraphicsPipelineBuilder(GraphicsPipelineBuilder&&) = default;
        GraphicsPipelineBuilder& operator=(const GraphicsPipelineBuilder&) = delete;
        GraphicsPipelineBuilder& operator=(GraphicsPipelineBuilder&&) = default;

        GraphicsPipelineBuilder&& AddShaderModule(mvk::ShaderModule&& shaderModule)&&;
        GraphicsPipelineBuilder&& SetVertexInput(const mvk::VertexInputState& vertexInput)&&;
        GraphicsPipelineBuilder&& SetVertexInput(mvk::VertexInputState&& vertexInput)&&;
        GraphicsPipelineBuilder&& SetPrimitiveTopology(vk::PrimitiveTopology topology)&&;
        GraphicsPipelineBuilder&& SetPrimitiveRestart(bool primitiveRestartEnable)&&;
        GraphicsPipelineBuilder&& SetCullMode(vk::CullModeFlags cullMode)&&;
        GraphicsPipelineBuilder&& SetInputAssembly(vk::PipelineInputAssemblyStateCreateInfo assembly)&&;
        
        GraphicsPipelineBuilder&& SetViewportState(vk::PipelineViewportStateCreateInfo viewportState)&&;
        GraphicsPipelineBuilder&& SetRasterizer(vk::PipelineRasterizationStateCreateInfo rasterizer)&&;
        GraphicsPipelineBuilder&& SetMultisample(vk::PipelineMultisampleStateCreateInfo multisample)&&;
        GraphicsPipelineBuilder&& SetDepthStencil(vk::PipelineDepthStencilStateCreateInfo depthStencil)&&;
        GraphicsPipelineBuilder&& SetDepthState(bool testEnable, bool writeEnable, vk::CompareOp compareOp, const std::optional<std::pair<float, float>>& depthBounds = {})&&;
        GraphicsPipelineBuilder&& SetColorBlend(const ColorBlendState& blend)&&;
        GraphicsPipelineBuilder&& SetColorBlend(ColorBlendState&& blend)&&;
        GraphicsPipelineBuilder&& SetPipelineLayout(mvk::PipelineLayout&& layout)&&;
        GraphicsPipelineBuilder&& AetDynamicState(vk::DynamicState dynamicState)&&;
        GraphicsPipelineBuilder&& SetDynamicStates(std::vector<vk::DynamicState>&& dynamicStates)&&;

        GraphicsPipelineBuilder&& AddColorAttachment(vk::Format format)&&;
        GraphicsPipelineBuilder&& SetColorAttachments(const std::vector<vk::Format>& formats)&&;
        GraphicsPipelineBuilder&& SetDepthAttachment(vk::Format format)&&;
        GraphicsPipelineBuilder&& SetStencilAttachment(vk::Format format)&&;
        GraphicsPipelineBuilder&& SetDepthNStencilAttachment(vk::Format format)&&;
        GraphicsPipelineBuilder&& SetRenderingInfo(const vk::PipelineRenderingCreateInfo& renderingInfo)&&;

        [[deprecated("Use SetColorAttachmentFormats instead")]]
        GraphicsPipelineBuilder&& SetRenderPass(vk::RenderPass renderPass, uint32_t subpass = 0)&&;

        GraphicsPipelineBuilder&& SetBasePipeline(vk::Pipeline basePipeline)&&;
        GraphicsPipelineBuilder&& SetBasePipelineIndex(int32_t index)&&;
        GraphicsPipelineBuilder&& AllowDerivatives()&&;
        GraphicsPipelineBuilder&& SetAsDerivative()&&;

        GraphicsPipeline Build()&&;

    private:
        std::string m_name;

        vk::PipelineCreateFlags m_flags = {};
        std::vector<mvk::ShaderModule> m_shaderModules;
        std::optional<mvk::VertexInputState> m_vertexStateInput;
        std::optional<vk::PipelineInputAssemblyStateCreateInfo> m_inputAssembly;
        std::optional<vk::PipelineTessellationStateCreateInfo> m_tessellation;
        vk::PipelineViewportStateCreateInfo m_viewportState;
        vk::PipelineRasterizationStateCreateInfo m_rasterizer;
        vk::PipelineMultisampleStateCreateInfo m_multisample;
        vk::PipelineDepthStencilStateCreateInfo m_depthStencil;
        std::optional<mvk::ColorBlendState> m_colorBlend;
        std::optional<mvk::PipelineLayout> m_layout;
        vk::RenderPass m_renderPass = VK_NULL_HANDLE;
        uint32_t m_subpass = 0;
        vk::Pipeline m_basePipeline = VK_NULL_HANDLE;
        int32_t m_basePipelineIndex = -1;

        std::vector<vk::DynamicState> m_dynamicStates;
        std::vector<vk::Format> m_colorAttachmentFormats;
        vk::Format m_depthAttachmentFormat = vk::Format::eUndefined;
        vk::Format m_stencilAttachmentFormat = vk::Format::eUndefined;
        std::optional<vk::PipelineRenderingCreateInfo> m_renderingInfo;

        void BuildRenderingInfo();
        bool IsUsingDynamicRendering() const;
        void ValidateConfiguration() const;
    };
}

#include "HeaderPost.h"