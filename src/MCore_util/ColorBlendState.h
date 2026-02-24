#pragma once

#include "VulkanDef.h"

#include <glm/glm.hpp>

#include "HeaderPre.h"

namespace mvk
{
    class __MY_EXT_CLASS__ ColorBlendState
    {
    public:
        ColorBlendState() = delete;
        ColorBlendState(
            vk::PipelineColorBlendStateCreateFlags flags,
            vk::Bool32 logicOpEnable,
            vk::LogicOp logicOp,
            std::vector<vk::PipelineColorBlendAttachmentState>&& attachments,
            glm::vec4 blendConstants);
        ColorBlendState(const ColorBlendState&);
        ColorBlendState(ColorBlendState&&) noexcept;
        ColorBlendState& operator=(const ColorBlendState&);
        ColorBlendState& operator=(ColorBlendState&&) noexcept;

        const vk::PipelineColorBlendStateCreateInfo& Get() const { return m_info; }
        operator vk::PipelineColorBlendStateCreateInfo() const { return m_info; }

    private:
        vk::PipelineColorBlendStateCreateInfo m_info;
        std::vector<vk::PipelineColorBlendAttachmentState> m_attachments;
        glm::vec4 m_blendConstants;
    };

    class __MY_EXT_CLASS__ ColorBlendStateBuilder
    {
    public:
        ColorBlendStateBuilder() = default;
        ColorBlendStateBuilder(const ColorBlendStateBuilder&) = delete;
        ColorBlendStateBuilder(ColorBlendStateBuilder&&) noexcept = default;
        ColorBlendStateBuilder& operator=(const ColorBlendStateBuilder&) = delete;
        ColorBlendStateBuilder& operator=(ColorBlendStateBuilder&&) noexcept = default;

        ColorBlendStateBuilder&& SetFlags(vk::PipelineColorBlendStateCreateFlags flags)&&;
        ColorBlendStateBuilder&& SetLogicOp(vk::Bool32 enable, vk::LogicOp op = vk::LogicOp::eCopy)&&;
        ColorBlendStateBuilder&& SetBlendConstants(float r, float g, float b, float a)&&;
        ColorBlendStateBuilder&& SetBlendConstants(const glm::vec4& constants)&&;

        ColorBlendStateBuilder&& AddAttachment(const vk::PipelineColorBlendAttachmentState& attachment)&&;
        ColorBlendStateBuilder&& AddAttachments(const std::vector<vk::PipelineColorBlendAttachmentState>& attachments)&&;

        ColorBlendStateBuilder&& AddDisabledAttachment()&&;
        ColorBlendStateBuilder&& AddAlphaBlendAttachment()&&;
        ColorBlendStateBuilder&& AddAdditiveBlendAttachment()&&;
        ColorBlendStateBuilder&& AddSubtractiveBlendAttachment()&&;
        ColorBlendStateBuilder&& AddMultiplyBlendAttachment()&&;
        ColorBlendStateBuilder&& AddScreenBlendAttachment()&&;

        ColorBlendStateBuilder&& AddCustomAttachment(
            vk::Bool32 blendEnable,
            vk::BlendFactor srcColorBlendFactor,
            vk::BlendFactor dstColorBlendFactor,
            vk::BlendOp colorBlendOp,
            vk::BlendFactor srcAlphaBlendFactor,
            vk::BlendFactor dstAlphaBlendFactor,
            vk::BlendOp alphaBlendOp,
            vk::ColorComponentFlags colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)&&;

        mvk::ColorBlendState Build()&&;

    private:
        vk::PipelineColorBlendStateCreateFlags m_flags{};
        vk::Bool32 m_logicOpEnable = VK_FALSE;
        vk::LogicOp m_logicOp = vk::LogicOp::eCopy;
        std::vector<vk::PipelineColorBlendAttachmentState> m_attachments;
        glm::vec4 m_blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f };
    };
}

#include "HeaderPost.h"