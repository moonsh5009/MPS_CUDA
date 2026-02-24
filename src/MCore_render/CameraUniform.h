#pragma once

#include "../MCore_util/SignalSlot.h"

#include "RenderUniformBase.h"
#include "Camera.h"

#include "HeaderPre.h"

namespace mcore::render
{
    struct alignas(16) CameraHostData
    {
        alignas(16) glm::mat4 viewMat;
        alignas(16) glm::mat4 projMat;
        alignas(16) glm::mat4 viewInvMat;
        alignas(16) glm::mat4 projInvMat;
        alignas(16) glm::vec3 position;
        alignas(16) glm::vec3 viewDir;
        alignas(16) glm::ivec4 viewport;
        alignas(8) glm::vec2 frustum;
    };

    class __MY_EXT_CLASS__ CameraUniform : public RenderUniformBase<CameraHostData>, public SignalSlot
    {
        DECLARE_RENDER_UNIFORM(CameraUniform)

    public:
        CameraUniform(IScene* pScene);

        void Initialize() override;
        bool Update(const vk::CommandBuffer& commandBuffer) override;

		void SetZoomFit() { m_bZoomFit = true; }

        Camera& GetCamera() { return m_camera; }
        const Camera& GetCamera() const { return m_camera; }

    private:
        bool m_bZoomFit;
        Camera m_camera;
    };
}

#include "HeaderPost.h"