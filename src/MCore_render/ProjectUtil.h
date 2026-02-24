#pragma once

#include <glm/glm.hpp>

namespace mcore
{
    class ProjectUtil final
    {
    public:
        static constexpr glm::mat4 Ortho(
            float xLeft, float xRight,
            float yBottom, float yTop,
            float zNear, float zFar) noexcept
        {
            return
            {
                2.f / (xRight - xLeft), 0.f, 0.f, 0.f,
                0.f, 2.f / (yTop - yBottom), 0.f, 0.f,
                0.f, 0.f, 1.f / (zFar - zNear), 0.f,
                -(xRight + xLeft) / (xRight - xLeft), -(yTop + yBottom) / (yTop - yBottom), zFar / (zFar - zNear), 1.f
            };
        }

        static constexpr glm::mat4 Frustum(
            float xLeft, float xRight,
            float yBottom, float yTop,
            float zNear, float zFar) noexcept
        {
            return
            {
                (2.f * zNear) / (xRight - xLeft), 0.f, 0.f, 0.f,
                0.f, (2.f * zNear) / (yTop - yBottom), 0.f, 0.f,
                (xRight + xLeft) / (xRight - xLeft), (yTop + yBottom) / (yTop - yBottom), zNear / (zFar - zNear), -1.f,
                0.f, 0.f, (zFar * zNear) / (zFar - zNear), 0.f
            };
        }

        static constexpr glm::vec3 Project(
            const glm::vec3& worldPos, const glm::mat4& viewMatrix,
            const glm::mat4& projMatrix, const glm::ivec4& viewport) noexcept
        {
            glm::vec4 retVal = { worldPos, 1. };
            retVal = (viewMatrix * retVal);
            retVal = (projMatrix * retVal);
            retVal /= retVal.w;

            retVal.x = (retVal.x + 1.f) * .5f;
            retVal.y = (retVal.y + 1.f) * .5f;

            retVal.x = (retVal.x * viewport[2]) + viewport[0];
            retVal.y = (retVal.y * viewport[3]) + viewport[1];
            return retVal;
        }

        static constexpr glm::dvec3 Unproject(
            const glm::vec3& screenPos, const glm::mat4& viewInvMatrix,
            const glm::mat4& projInvMatrix, const glm::ivec4& viewport) noexcept
        {
            glm::vec4 retVal = { screenPos, 1.f };
            retVal.x = (retVal.x - viewport[0]) / viewport[2];
            retVal.y = (retVal.y - viewport[1]) / viewport[3];

            retVal.x = retVal.x * 2.f - 1.f;
            retVal.y = retVal.y * 2.f - 1.f;

            retVal = viewInvMatrix * projInvMatrix * retVal;
            retVal /= retVal.w;
            return retVal;
        }
    };
}