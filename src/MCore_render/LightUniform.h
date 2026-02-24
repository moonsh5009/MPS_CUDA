#pragma once

#include "RenderUniformBase.h"

#include "HeaderPre.h"

namespace mcore::render
{
    struct alignas(16) LightColor
    {
        alignas(16) glm::vec4 ambient;
        alignas(16) glm::vec4 diffuse;
        alignas(16) glm::vec4 specular;
    };

    struct alignas(16) LightHostData
    {
        alignas(16) glm::vec4 pos;
        alignas(16) LightColor color;
        alignas(16) glm::mat4 shadow[4];
        alignas(16) glm::vec4 layer_distance;
        alignas(4) int layer_num;
        alignas(4) int is_shadow_map;
    };

	class __MY_EXT_CLASS__ LightUniform : public RenderUniformBase<LightHostData>
    {
        DECLARE_RENDER_UNIFORM(LightUniform)

    public:
        LightUniform(IScene* pScene);

        void Initialize() override;
        bool Update(const vk::CommandBuffer& commandBuffer) override;

    private:
		LightHostData m_hostData;
    };
}

#include "HeaderPost.h"