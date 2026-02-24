#include "stdafx.h"
#include "CameraUniform.h"

#include "../MCore_interface/IScene.h"

#include "CameraUtil.h"

using namespace mcore::render;

IMPLEMENT_RENDER_UNIFORM(CameraUniform, RenderUniformType::CAMERA, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute)

CameraUniform::CameraUniform(IScene* pScene)
    : RenderUniformBase{ pScene }
	, m_bZoomFit{ true }
{}

void CameraUniform::Initialize()
{
    m_camera.Initialize();
    Bind(m_camera.onUpdateMatrix, [&]
    {
        SetDirty();
    });

    Bind(GetScene()->onUpdateAABB, [&](const AABBf& aabb)
    {
        SetDirty();
    });

    SetDirty();
    m_bZoomFit = true;
}

bool CameraUniform::Update(const vk::CommandBuffer& commandBuffer)
{
    if (!IsDirty())
        return false;

    if (m_bZoomFit)
    {
        m_bZoomFit = false;

        auto aabbCopy = GetScene()->GetAABB();
        if (m_camera.GetProjection().GetProjectionType() == ProjectionType::ORTHO)
            aabbCopy.Scale(3.0f);
        else
            aabbCopy.Scale(1.2f);
        CameraUtil::FitToAABB(m_camera, aabbCopy);
    }

    CameraUtil::FitNearFarToAABB(m_camera, GetScene()->GetAABB());
    m_camera.UpdateMatrix(false);

    const auto viewport = GetScene()->GetViewport();
    const auto& transform = m_camera.GetTransform();
    const auto& projInfo = m_camera.GetProjection();
    // WebGPU to Vulkan conversion matrix (Y-flip)
    // Vulkan has inverted Y-axis compared to WebGPU
    static constexpr glm::mat4 webgpuToVulkan = glm::mat4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );

    CameraHostData cameraHostBuffer = {
        .viewMat = transform.GetViewMatrix(),
        .projMat = webgpuToVulkan * projInfo.getProjectionMatrix(),
        .viewInvMat = transform.GetViewInvMatrix(),
        .projInvMat = webgpuToVulkan * projInfo.getProjectionInvMatrix(),
        .position = transform.GetPosition(),
        .viewDir = glm::normalize(transform.GetForward()),
        .viewport = viewport,
        .frustum = { projInfo.GetNear(), projInfo.GetFar() },
    };
    GetUBO().CopyFromHostAndUpload(commandBuffer, cameraHostBuffer);
    
    ClearDirty();
    return true;
}