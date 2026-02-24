#pragma once

#include "../MCore_util/AABB.h"
#include "../MCore_util/Quaternion.h"
#include "Camera.h"

#include "HeaderPre.h"

namespace mcore::render
{
    class __MY_EXT_CLASS__ CameraUtil final
    {
    private:
        // return value: { near, far, width, height }
        static glm::vec4 GetAABBProjectionInfo(const Camera& camera, const AABBf& aabb) noexcept;

    public:
        static void FitToAABB(Camera& camera, const AABBf& aabb, const glm::vec3& forward, const glm::vec3& up = { 0., 0., 1. }) noexcept;
        static void FitToAABB(Camera& camera, const AABBf& aabb) noexcept;
        static void FitNearFarToAABB(Camera& camera, const AABBf& aabb) noexcept;

        static glm::vec3 Unproject(const Camera& camera, const glm::vec3& screenPos, const glm::ivec4& viewport) noexcept;

        // worldSpaceDepthAnchor의 sreeen depth를 추출하여 사용합니다.
        static glm::vec3 Unproject(
            const Camera& camera, const glm::vec2& screenPos,
            const glm::vec3& worldSpaceDepthAnchor, const glm::ivec4& viewport) noexcept;

        static glm::vec3 Project(const Camera& camera, const glm::vec3& worldPos, const glm::ivec4& viewport) noexcept;

        static glm::vec3 RayProjectFromScreenOntoAABB(
            const Camera& camera, const AABBf& aabb, const glm::vec2& screenPos, const glm::ivec4& viewport) noexcept;

        static std::string to_string(const Camera& camera) noexcept;
        static float GetPixelSize(const Camera& camera, float screenHeight) noexcept;

        static std::pair<Quaternionf, float> ConvertParamsLegacyToCurrent(
            const glm::vec3& eyeDir, const glm::vec3& upDir, float focalDistance) noexcept;
    };
}

#include "HeaderPost.h"