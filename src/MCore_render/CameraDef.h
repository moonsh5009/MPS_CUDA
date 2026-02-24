#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace mcore::render
{
    enum class ProjectionType : unsigned
    {
        ORTHO = 0,
        PERSPECTIVE,
        NONE,
    };

    namespace constant
    {
        namespace camera
        {
            constexpr ProjectionType DEFAULT_PROJ_TYPE = ProjectionType::ORTHO;

            constexpr float MIN_SCALE = .001f;
            constexpr float MAX_SCALE = 1000.f;

            constexpr float DEFAULT_HEIGHT = .5f;
            constexpr float MAX_HEIGHT = 300000.f;
            constexpr float MIN_HEIGHT = .001f;

            constexpr float DEFAULT_ASPECT_RATIO = 1.f;

            constexpr float DEFAULT_Z_NEAR = .5f;
            constexpr float MIN_Z_NEAR = .01f;

            constexpr float DEFAULT_Z_FAR = 500.f;
            constexpr float MAX_Z_FAR = 1000000.f;

            constexpr float DEFAULT_FOV = (glm::quarter_pi<float>() * .8f);
        }
    }
}