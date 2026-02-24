#include "stdafx.h"
#include "OpaqueRenderer.h"

#include "../MCore_interface/IRenderCore.h"

#include "MSAARenderTarget.h"

using namespace mcore;
using namespace mcore::render;

OpaqueRenderer::OpaqueRenderer(IRenderingEngine* pRenderingEngine)
    : RendererBase{ pRenderingEngine, "[Renderer] Opaque" }
{}

const mvk::GraphicsPipeline& OpaqueRenderer::LoadPointPipeline()
{
    if (m_pointPipeline)
        return m_pointPipeline;

    const auto pScene = GetRenderingEngine()->GetScene();
    const auto pRenderCore = pScene->GetRenderCore();

    mvk::ShaderModule vs{ "render/common/point.vert" };
    mvk::ShaderModule fs{ "render/common/point.frag" };
    vs.SetStage(vk::ShaderStageFlagBits::eVertex);
    fs.SetStage(vk::ShaderStageFlagBits::eFragment);

    auto layout = mvk::PipelineLayoutBuilder()
        .AddBindGroupLayout(0, pRenderCore->GetUniformBindGrouplayout())
        .AddBindGroupLayout(1, pRenderCore->GetPointStreamBindGrouplayout())
        .Build();

    auto vertexInputState = mvk::VertexInputStateBuilder{}
        .Build();

    auto colorBlandState = mvk::ColorBlendStateBuilder{}
        .AddAttachment({
            vk::False,
            vk::BlendFactor::eOne,
            vk::BlendFactor::eOne,
            vk::BlendOp::eAdd,
            vk::BlendFactor::eZero,
            vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags
        })
        .Build();

    m_pointPipeline = mvk::GraphicsPipelineBuilder("opaque point")
        .AddShaderModule(std::move(vs))
        .AddShaderModule(std::move(fs))
        .SetPipelineLayout(std::move(layout))
        .SetVertexInput(std::move(vertexInputState))
        .SetColorBlend(std::move(colorBlandState))
        .SetDepthState(true, true, vk::CompareOp::eGreater)
        .SetPrimitiveTopology(vk::PrimitiveTopology::eTriangleList)
        .SetDynamicStates({ vk::DynamicState::eViewportWithCount, vk::DynamicState::eScissorWithCount })
        .SetMultisample(pRenderCore->GetMultiSampleState())
        .AddColorAttachment(pRenderCore->GetColorFormat())
        .SetDepthAttachment(pRenderCore->GetDepthFormat())
        .Build();

    return m_pointPipeline;
}

const mvk::GraphicsPipeline& OpaqueRenderer::LoadLinePipeline()
{
    if (m_linePipeline)
        return m_linePipeline;

    const auto pScene = GetRenderingEngine()->GetScene();
    const auto pRenderCore = pScene->GetRenderCore();

    mvk::ShaderModule vs{ "render/common/line.vert" };
    mvk::ShaderModule fs{ "render/common/line.frag" };
    vs.SetStage(vk::ShaderStageFlagBits::eVertex);
    fs.SetStage(vk::ShaderStageFlagBits::eFragment);

    auto layout = mvk::PipelineLayoutBuilder()
        .AddBindGroupLayout(0, pRenderCore->GetUniformBindGrouplayout())
        .AddBindGroupLayout(1, pRenderCore->GetLineStreamBindGrouplayout())
        .Build();

    auto vertexInputState = mvk::VertexInputStateBuilder{}
    .Build();

    auto colorBlandState = mvk::ColorBlendStateBuilder{}
        .AddAttachment({
        vk::False,
        vk::BlendFactor::eOne,
        vk::BlendFactor::eOne,
        vk::BlendOp::eAdd,
        vk::BlendFactor::eZero,
        vk::BlendFactor::eZero,
        vk::BlendOp::eAdd,
        vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags
        })
        .Build();

    m_linePipeline = mvk::GraphicsPipelineBuilder("opaque line")
        .AddShaderModule(std::move(vs))
        .AddShaderModule(std::move(fs))
        .SetPipelineLayout(std::move(layout))
        .SetVertexInput(std::move(vertexInputState))
        .SetColorBlend(std::move(colorBlandState))
        .SetDepthState(true, true, vk::CompareOp::eGreater)
        .SetPrimitiveTopology(vk::PrimitiveTopology::eTriangleList)
        .SetDynamicStates({ vk::DynamicState::eViewportWithCount, vk::DynamicState::eScissorWithCount })
        .SetMultisample(pRenderCore->GetMultiSampleState())
        .AddColorAttachment(pRenderCore->GetColorFormat())
        .SetDepthAttachment(pRenderCore->GetDepthFormat())
        .Build();

    return m_linePipeline;
}

const mvk::GraphicsPipeline& OpaqueRenderer::LoadTrianglePipeline()
{
    if (m_trianglePipeline)
        return m_trianglePipeline;

	const auto pScene = GetRenderingEngine()->GetScene();
	const auto pRenderCore = pScene->GetRenderCore();

    mvk::ShaderModule vs{ "render/common/triangle.vert" };
    mvk::ShaderModule fs{ "render/common/triangle.frag" };
    vs.SetStage(vk::ShaderStageFlagBits::eVertex);
    fs.SetStage(vk::ShaderStageFlagBits::eFragment);

    auto layout = mvk::PipelineLayoutBuilder()
        .AddBindGroupLayout(0, pRenderCore->GetUniformBindGrouplayout())
        .Build();

    auto vertexInputState = mvk::VertexInputStateBuilder{}
        .AddBinding(0, sizeof(mcore::Vector3))
        .AddDouble3Attribute(0, 0, 0)
        .AddBinding(1, sizeof(mcore::Vector3))
        .AddDouble3Attribute(1, 2, 0)
        .AddBinding(2, sizeof(mvk::vbo::TriangleAttribute))
        .AddColorAttribute(2, 4, offsetof(mvk::vbo::TriangleAttribute, color))
        .AddFloat2Attribute(2, 5, offsetof(mvk::vbo::TriangleAttribute, texCoord))
        .Build();

    auto colorBlandState = mvk::ColorBlendStateBuilder{}
        .AddAttachment({
            vk::False,
            vk::BlendFactor::eOne,
            vk::BlendFactor::eOne,
            vk::BlendOp::eAdd,
            vk::BlendFactor::eZero,
            vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags
        })
        .Build();

    m_trianglePipeline = mvk::GraphicsPipelineBuilder("opaque triangle")
        .AddShaderModule(std::move(vs))
        .AddShaderModule(std::move(fs))
        .SetPipelineLayout(std::move(layout))
        .SetVertexInput(std::move(vertexInputState))
        .SetColorBlend(std::move(colorBlandState))
        .SetDepthState(true, true, vk::CompareOp::eGreater)
        .SetPrimitiveTopology(vk::PrimitiveTopology::eTriangleList)
        .SetDynamicStates({ vk::DynamicState::eViewportWithCount, vk::DynamicState::eScissorWithCount })
        .SetMultisample(pRenderCore->GetMultiSampleState())
        .AddColorAttachment(pRenderCore->GetColorFormat())
        .SetDepthAttachment(pRenderCore->GetDepthFormat())
        .Build();

    return m_trianglePipeline;
}