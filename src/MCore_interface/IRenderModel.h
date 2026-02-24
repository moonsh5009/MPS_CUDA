#pragma once

#include "RenderTypeDef.h"
#include "StreamBuffer.h"

#include <memory>

namespace mcore
{
	class IScene;
	class IRenderModelContainer;
	class IRenderModel
	{
	public:
		IRenderModel() = delete;
		IRenderModel(IRenderModelContainer* pModelContainer)
			: m_pModelContainer{ pModelContainer }
			, m_bShow{ true }
		{}
		virtual ~IRenderModel() = default;
		IRenderModel(const IRenderModel&) = default;
		IRenderModel(IRenderModel&&) = default;
		IRenderModel& operator=(const IRenderModel&) = default;
		IRenderModel& operator=(IRenderModel&&) = default;

		virtual void Initialize(const vk::CommandBuffer& commandBuffer) = 0;
		virtual void PreProcess(IScene* pScene, const vk::CommandBuffer& commandBuffer) = 0;

		virtual render::RenderModelType GetType() const = 0;
		virtual uint32_t GetBindGroupInstance() const = 0;

		IRenderModelContainer* GetModelContainer() const { return m_pModelContainer; }

		PointStreamBuffer& GetPointStream() { return m_pointStream; }
		LineStreamBuffer& GetLineStream() { return m_lineStream; }
		TriangleStreamBuffer& GetTriangleStream() { return m_triangleStream; }

		const PointStreamBuffer& GetPointStream() const { return m_pointStream; }
		const LineStreamBuffer& GetLineStream() const { return m_lineStream; }
		const TriangleStreamBuffer& GetTriangleStream() const { return m_triangleStream; }

		bool GetShow() const { return m_bShow; }
		void SetShow(bool bShow) { m_bShow = bShow; }

	protected:
		IRenderModelContainer* m_pModelContainer;

		PointStreamBuffer m_pointStream;
		LineStreamBuffer m_lineStream;
		TriangleStreamBuffer m_triangleStream;

		bool m_bShow;
	};

	template <typename T>
	concept DerivedRenderModel = std::derived_from<T, IRenderModel>;

	using RenderModelArray = std::array<std::unique_ptr<IRenderModel>, static_cast<size_t>(render::RenderModelType::Size)>;
}