#include "stdafx.h"
#include "CameraUtil.h"

#include "ProjectUtil.h"

#include <sstream>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/intersect.hpp>

using namespace std;
using namespace glm;

namespace
{
    constexpr float NEAR_FAR_MARGIN = 1.0;
}

namespace mcore::render
{
    vec4 CameraUtil::GetAABBProjectionInfo(const Camera& camera, const AABBf& aabb) noexcept
    {
        const auto& transform = camera.GetTransform();

        const auto& cameraPos = transform.GetPosition();
        const auto& cameraForward = transform.GetForward();
        const auto& cameraHorizontal = transform.GetHorizontal();
        const auto& cameraVertical = transform.GetVertical();

        const auto& aabbMin = aabb.GetMin();
        const auto& aabbMax = aabb.GetMax();

        auto aabbNear = numeric_limits<float>::max();
        auto aabbFar = numeric_limits<float>::lowest();
        auto width = 0.f;
        auto height = 0.f;

        for (const auto aabbVertZ : { aabbMin.z, aabbMax.z })
        {
            for (const auto aabbVertY : { aabbMin.y, aabbMax.y })
            {
                for (const auto aabbVertX : { aabbMin.x, aabbMax.x })
                {
                    const vec3 aabbVert = { aabbVertX , aabbVertY , aabbVertZ };
                    const vec4 aabbVertToCameraPos = { cameraPos - aabbVert, 1.f };

                    const auto displacement = glm::dot(cameraForward, aabbVertToCameraPos);
                    const auto proj_aabbVertToCameraPos = (displacement * cameraForward);
                    const auto perp_aabbVertToCameraPos = (aabbVertToCameraPos - proj_aabbVertToCameraPos);

                    aabbNear = glm::min(aabbNear, displacement);
                    aabbFar = glm::max(aabbFar, displacement);
                    width = glm::max(width, glm::abs(glm::dot(cameraHorizontal, perp_aabbVertToCameraPos)));
                    height = glm::max(height, glm::abs(glm::dot(cameraVertical, perp_aabbVertToCameraPos)));
                }
            }
        }

        return { aabbNear, aabbFar, width, height };
    }

    void CameraUtil::FitToAABB(Camera& camera, const AABBf& aabb, const vec3& forward, const vec3& up) noexcept
    {
        CameraTransform& transform = camera.GetTransform();
        const vec3& normalizedForward = normalize(forward);

        vec3 validUp;
        if (epsilonEqual(abs(dot(normalizedForward, normalize(up))), 1.f, epsilon<float>()))
        {
            validUp =
            {
                normalizedForward.y - normalizedForward.z,
                normalizedForward.z - normalizedForward.x,
                normalizedForward.x - normalizedForward.y
            };
        }
        else
            validUp = up;

        transform.Orient(normalizedForward, validUp);
        transform.UpdateMatrix(false);

        FitToAABB(camera, aabb);
    }

    void CameraUtil::FitToAABB(Camera& camera, const AABBf& aabb) noexcept
    {
        CameraTransform& transform = camera.GetTransform();
        CameraProjection& projection = camera.GetProjection();

        const vec3& aabbCenter = aabb.GetCenter();
        const vec3& cameraForward = transform.GetForward();

        // set dummy position
        transform.SetPosition(aabbCenter + cameraForward);

        const vec4& aabbProjInfo = GetAABBProjectionInfo(camera, aabb);

        const float aabbNear = (aabbProjInfo[0] - NEAR_FAR_MARGIN);
        const float aabbFar = (aabbProjInfo[1] + NEAR_FAR_MARGIN);
        const float width = aabbProjInfo[2];
        const float height = aabbProjInfo[3];

        constexpr float HEIGHT_MARGIN_RATIO = 1.1f;
        const float fittedHeight = (glm::max(height, width / projection.GetAspectRatio()) * HEIGHT_MARGIN_RATIO);

        if (projection.GetProjectionType() == ProjectionType::ORTHO)
        {
            const float displacement = -aabbNear;

            transform.MoveForward(displacement);
            projection.SetNear(constant::camera::MIN_Z_NEAR);
            projection.SetHeight(fittedHeight);
            projection.SetFar(aabbFar + displacement);
        }
        else
        {
            const float tanHalfFov = tanf(constant::camera::DEFAULT_FOV * .5f);
            const float fovSynchronizedNear = (fittedHeight / tanHalfFov);
            const float displacement = (fovSynchronizedNear - aabbNear);

            transform.MoveForward(displacement);
            projection.SetNear(fovSynchronizedNear);
            projection.SetHeight(projection.GetNear() * tanHalfFov);
            projection.SetFar(aabbFar + displacement);
        }
    }

    void CameraUtil::FitNearFarToAABB(Camera& camera, const AABBf& aabb) noexcept
    {
        CameraTransform& transform = camera.GetTransform();
        CameraProjection& projection = camera.GetProjection();

        const vec3& cameraPos = transform.GetPosition();
        const vec3& cameraForward = transform.GetForward();

        const vec3& aabbMin = aabb.GetMin();
        const vec3& aabbMax = aabb.GetMax();

        const vec4& aabbProjInfo = GetAABBProjectionInfo(camera, aabb);

        const float aabbNear = (aabbProjInfo[0] - NEAR_FAR_MARGIN);
        const float aabbFar = (aabbProjInfo[1] + NEAR_FAR_MARGIN);

        if (projection.GetProjectionType() == ProjectionType::ORTHO)
        {
            const float displacement = -aabbNear;

            transform.MoveForward(displacement);
            projection.SetFar(aabbFar + displacement);
        }
        else
        {
            const float tanHalfFov = tanf(constant::camera::DEFAULT_FOV * .5f);

            projection.SetNear(aabbNear);
            projection.SetHeight(projection.GetNear() * tanHalfFov);
            projection.SetFar(aabbFar);
        }
    }

    vec3 CameraUtil::Unproject(const Camera& camera, const vec3& screenPos, const ivec4& viewport) noexcept
    {
        const CameraTransform& transform = camera.GetTransform();
        const CameraProjection& projection = camera.GetProjection();

        return ProjectUtil::Unproject(screenPos, transform.GetViewInvMatrix(), projection.getProjectionInvMatrix(), viewport);
    }

    vec3 CameraUtil::Unproject(const Camera& camera, const vec2& screenPos, const vec3& worldSpaceDepthAnchor, const ivec4& viewport) noexcept
    {
        const vec3& screenSpaceAnchor = Project(camera, worldSpaceDepthAnchor, viewport);
        return Unproject(camera, { screenPos, screenSpaceAnchor.z }, viewport);
    }

    vec3 CameraUtil::RayProjectFromScreenOntoAABB(
        const Camera& camera, const AABBf& aabb, const vec2& screenPos, const ivec4& viewport) noexcept
    {
        const CameraTransform& transform = camera.GetTransform();
        const CameraProjection& projection = camera.GetProjection();

        vec3 rayPos{};
        vec3 rayDirection{};
        vec3 aabbCenter = aabb.GetCenter();

        // Projection에 따라 screenPos로부터의 view ray를 계산한다.

        if (projection.GetProjectionType() == ProjectionType::ORTHO)
        {
            rayPos = Unproject(camera, { screenPos, 0. }, viewport);
            rayDirection = -transform.GetForward();

            float stepLength = 0.;
            const auto _ = intersectRayPlane(rayPos, rayDirection, aabbCenter, -rayDirection, stepLength);
            return (rayPos + (rayDirection * abs(stepLength)));
        }

        // Perspective mode
        vec3 targetPos = Unproject(camera, { screenPos, 0. }, viewport);

        float stepLength = 0.;
        const auto _ = intersectRayPlane(rayPos, rayDirection, aabbCenter, -rayDirection, stepLength);
        auto delta = abs(stepLength);

        targetPos.z += delta;
        return targetPos;
    }

    vec3 CameraUtil::Project(const Camera& camera, const vec3& worldPos, const ivec4& viewport) noexcept
    {
        const CameraTransform& transform = camera.GetTransform();
        const CameraProjection& projection = camera.GetProjection();

        return ProjectUtil::Project(worldPos, transform.GetViewMatrix(), projection.getProjectionMatrix(), viewport);
    }

    string CameraUtil::to_string(const Camera& camera) noexcept
    {
        const CameraTransform& transform = camera.GetTransform();
        const CameraProjection& projection = camera.GetProjection();

        const mat4& viewMat = transform.GetViewMatrix();
        const mat4& projMat = projection.getProjectionMatrix();

        stringstream ss;
        ss << "* viewMatrix\n"
            << glm::to_string(viewMat) << std::endl;
        ss << "* projMatrix\n"
            << glm::to_string(projMat) << std::endl;

        ss << "* view info" << std::endl;
        ss << "position\n"
            << glm::to_string(transform.GetPosition()) << std::endl;
        ss << "dir\n"
            << glm::to_string(transform.GetForward()) << std::endl;
        ss << "up\n"
            << glm::to_string(transform.GetVertical()) << std::endl;

        ss << "* proj info" << std::endl;
        ss << "HalfHeight\n"
            << std::to_string(projection.GetHeight()) << std::endl;

        ss << "* screen info" << std::endl;

        return ss.str();
    }

    float CameraUtil::GetPixelSize(const Camera& camera, float screenHeight) noexcept
    {
        return (camera.GetProjection().GetHeight() * 2.f) / screenHeight;
    }

    pair<Quaternionf, float> CameraUtil::ConvertParamsLegacyToCurrent(
        const vec3& eyeDir, const vec3& upDir, float focalDistance) noexcept
    {
        Quaternionf rotation;
        rotation.Orient(eyeDir, upDir);

        const float height = (focalDistance * glm::tan(quarter_pi<float>() * .5f));

        return { rotation, height };
    }
}