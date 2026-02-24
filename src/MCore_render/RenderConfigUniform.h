#pragma once

#include "RenderUniformBase.h"

#include "HeaderPre.h"

namespace mcore::render
{
    enum class ColorType : uint32_t
    {
        Const = 0,
        Attribute = 1,
	};
    struct alignas(16) RenderConfigHostData
    {
        alignas(16) glm::vec4 const_color;
        alignas(4) ColorType color_type;
        alignas(4) ColorType transparent_type;

        alignas(4) uint32_t model_type;

		alignas(4) int use_point_size;
        alignas(4) float point_size;

        alignas(4) int use_line_width;
		alignas(4) float line_width;
    };

	class __MY_EXT_CLASS__ RenderConfigUniform : public RenderUniformBase<RenderConfigHostData>
    {
        DECLARE_RENDER_UNIFORM(RenderConfigUniform)

    public:
        RenderConfigUniform(IScene* pScene);

        void Initialize() override;
        bool Update(const vk::CommandBuffer& commandBuffer) override;

        void SetConstColor(const glm::vec4& color);
		void SetColorType(ColorType type);
		void SetTransparentType(ColorType type);

		void SetModelType(RenderModelType type);
        
        void SetUsePointSize(bool use);
        void SetPointSize(float size);

        void SetUseLineWidth(bool use);
        void SetLineWidth(float width);

    private:
		RenderConfigHostData m_hostData;
    };
}

#include "HeaderPost.h"