#include "stdafx.h"
#include "ColorBlendState.h"

mvk::ColorBlendState::ColorBlendState(
    vk::PipelineColorBlendStateCreateFlags flags,
    vk::Bool32 logicOpEnable,
    vk::LogicOp logicOp,
    std::vector<vk::PipelineColorBlendAttachmentState>&& attachments,
    glm::vec4 blendConstants)
    : m_attachments{ std::move(attachments) }
    , m_blendConstants{ blendConstants }
{
    m_info.flags = flags;
    m_info.logicOpEnable = logicOpEnable;
    m_info.logicOp = logicOp;
    m_info.attachmentCount = static_cast<uint32_t>(m_attachments.size());
    m_info.pAttachments = m_attachments.data();
    m_info.blendConstants[0] = m_blendConstants[0];
    m_info.blendConstants[1] = m_blendConstants[1];
    m_info.blendConstants[2] = m_blendConstants[2];
    m_info.blendConstants[3] = m_blendConstants[3];
}

mvk::ColorBlendState::ColorBlendState(const ColorBlendState& other)
{
    *this = other;
}

mvk::ColorBlendState::ColorBlendState(ColorBlendState&& other) noexcept
{
    *this = std::move(other);
}

mvk::ColorBlendState& mvk::ColorBlendState::operator=(const ColorBlendState& other)
{
    if (this != &other)
    {
        m_attachments = other.m_attachments;
        m_blendConstants = other.m_blendConstants;

        m_info.flags = other.m_info.flags;
        m_info.logicOpEnable = other.m_info.logicOpEnable;
        m_info.logicOp = other.m_info.logicOp;
        m_info.attachmentCount = static_cast<uint32_t>(m_attachments.size());
        m_info.pAttachments = m_attachments.data();
        m_info.blendConstants[0] = m_blendConstants[0];
        m_info.blendConstants[1] = m_blendConstants[1];
        m_info.blendConstants[2] = m_blendConstants[2];
        m_info.blendConstants[3] = m_blendConstants[3];
    }
    return *this;
}

mvk::ColorBlendState& mvk::ColorBlendState::operator=(ColorBlendState&& other) noexcept
{
    if (this != &other)
    {
        m_attachments = std::move(other.m_attachments);
        m_blendConstants = std::move(other.m_blendConstants);

        other.m_info.attachmentCount = 0;
        other.m_info.pAttachments = nullptr;

        m_info.flags = other.m_info.flags;
        m_info.logicOpEnable = other.m_info.logicOpEnable;
        m_info.logicOp = other.m_info.logicOp;
        m_info.attachmentCount = static_cast<uint32_t>(m_attachments.size());
        m_info.pAttachments = m_attachments.data();
        m_info.blendConstants[0] = m_blendConstants[0];
        m_info.blendConstants[1] = m_blendConstants[1];
        m_info.blendConstants[2] = m_blendConstants[2];
        m_info.blendConstants[3] = m_blendConstants[3];
    }
    return *this;
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::SetFlags(vk::PipelineColorBlendStateCreateFlags flags)&&
{
    m_flags = flags;
    return std::move(*this);
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::SetLogicOp(vk::Bool32 enable, vk::LogicOp op)&&
{
    m_logicOpEnable = enable;
    m_logicOp = op;
    return std::move(*this);
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::SetBlendConstants(float r, float g, float b, float a)&&
{
    m_blendConstants = { r, g, b, a };
    return std::move(*this);
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::SetBlendConstants(const glm::vec4& constants)&&
{
    m_blendConstants = constants;
    return std::move(*this);
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::AddAttachment(const vk::PipelineColorBlendAttachmentState& attachment)&&
{
    m_attachments.push_back(attachment);
    return std::move(*this);
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::AddAttachments(const std::vector<vk::PipelineColorBlendAttachmentState>& attachments)&&
{
    m_attachments.insert(m_attachments.end(), attachments.begin(), attachments.end());
    return std::move(*this);
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::AddDisabledAttachment()&&
{
    vk::PipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    attachment.blendEnable = VK_FALSE;
    m_attachments.push_back(attachment);
    return std::move(*this);
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::AddAlphaBlendAttachment()&&
{
    // Classic alpha blending: (srcColor * srcAlpha) + (dstColor * (1 - srcAlpha))
    return std::move(*this).AddCustomAttachment(
        VK_TRUE,
        vk::BlendFactor::eSrcAlpha,           // srcColorBlendFactor
        vk::BlendFactor::eOneMinusSrcAlpha,   // dstColorBlendFactor
        vk::BlendOp::eAdd,                    // colorBlendOp
        vk::BlendFactor::eOne,                // srcAlphaBlendFactor
        vk::BlendFactor::eZero,               // dstAlphaBlendFactor
        vk::BlendOp::eAdd                     // alphaBlendOp
    );
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::AddAdditiveBlendAttachment()&&
{
    // Additive blending: srcColor + dstColor
    return std::move(*this).AddCustomAttachment(
        VK_TRUE,
        vk::BlendFactor::eOne,     // srcColorBlendFactor
        vk::BlendFactor::eOne,     // dstColorBlendFactor
        vk::BlendOp::eAdd,         // colorBlendOp
        vk::BlendFactor::eOne,     // srcAlphaBlendFactor
        vk::BlendFactor::eZero,    // dstAlphaBlendFactor
        vk::BlendOp::eAdd          // alphaBlendOp
    );
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::AddSubtractiveBlendAttachment()&&
{
    // Subtractive blending: dstColor - srcColor
    return std::move(*this).AddCustomAttachment(
        VK_TRUE,
        vk::BlendFactor::eOne,     // srcColorBlendFactor
        vk::BlendFactor::eOne,     // dstColorBlendFactor
        vk::BlendOp::eSubtract,    // colorBlendOp
        vk::BlendFactor::eOne,     // srcAlphaBlendFactor
        vk::BlendFactor::eZero,    // dstAlphaBlendFactor
        vk::BlendOp::eAdd          // alphaBlendOp
    );
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::AddMultiplyBlendAttachment()&&
{
    // Multiply blending: srcColor * dstColor
    return std::move(*this).AddCustomAttachment(
        VK_TRUE,
        vk::BlendFactor::eDstColor,    // srcColorBlendFactor
        vk::BlendFactor::eZero,        // dstColorBlendFactor
        vk::BlendOp::eAdd,             // colorBlendOp
        vk::BlendFactor::eDstAlpha,    // srcAlphaBlendFactor
        vk::BlendFactor::eZero,        // dstAlphaBlendFactor
        vk::BlendOp::eAdd              // alphaBlendOp
    );
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::AddScreenBlendAttachment()&&
{
    // Screen blending: 1 - (1 - srcColor) * (1 - dstColor)
    return std::move(*this).AddCustomAttachment(
        VK_TRUE,
        vk::BlendFactor::eOneMinusDstColor,    // srcColorBlendFactor
        vk::BlendFactor::eOne,                 // dstColorBlendFactor
        vk::BlendOp::eAdd,                     // colorBlendOp
        vk::BlendFactor::eOneMinusDstAlpha,    // srcAlphaBlendFactor
        vk::BlendFactor::eOne,                 // dstAlphaBlendFactor
        vk::BlendOp::eAdd                      // alphaBlendOp
    );
}

mvk::ColorBlendStateBuilder&& mvk::ColorBlendStateBuilder::AddCustomAttachment(
    vk::Bool32 blendEnable,
    vk::BlendFactor srcColorBlendFactor,
    vk::BlendFactor dstColorBlendFactor,
    vk::BlendOp colorBlendOp,
    vk::BlendFactor srcAlphaBlendFactor,
    vk::BlendFactor dstAlphaBlendFactor,
    vk::BlendOp alphaBlendOp,
    vk::ColorComponentFlags colorWriteMask)&&
{
    vk::PipelineColorBlendAttachmentState attachment{};
    attachment.blendEnable = blendEnable;
    attachment.srcColorBlendFactor = srcColorBlendFactor;
    attachment.dstColorBlendFactor = dstColorBlendFactor;
    attachment.colorBlendOp = colorBlendOp;
    attachment.srcAlphaBlendFactor = srcAlphaBlendFactor;
    attachment.dstAlphaBlendFactor = dstAlphaBlendFactor;
    attachment.alphaBlendOp = alphaBlendOp;
    attachment.colorWriteMask = colorWriteMask;

    m_attachments.push_back(attachment);
    return std::move(*this);
}

mvk::ColorBlendState mvk::ColorBlendStateBuilder::Build()&&
{
    return ColorBlendState{
        m_flags,
        m_logicOpEnable,
        m_logicOp,
        std::move(m_attachments),
        m_blendConstants
    };
}