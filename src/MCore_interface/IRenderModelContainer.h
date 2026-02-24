#pragma once

#include "../MCore_util/AABB.h"
#include "../MCore_util/Signal.h"

#include "IRenderModel.h"

namespace mcore
{
	class IRenderCore;
	class IRenderModelContainer
	{
	public:
		IRenderModelContainer() = delete;
		IRenderModelContainer(IRenderCore* pRenderCore) : m_pRenderCore{ pRenderCore } {}
		virtual ~IRenderModelContainer() = default;
		IRenderModelContainer(const IRenderModelContainer&) = default;
		IRenderModelContainer(IRenderModelContainer&&) = default;
		IRenderModelContainer& operator=(const IRenderModelContainer&) = default;
		IRenderModelContainer& operator=(IRenderModelContainer&&) = default;

		virtual void Initialize() = 0;

		IRenderCore* GetRenderCore() const { return m_pRenderCore; }

	protected:
		IRenderCore* m_pRenderCore;

	public:
		template<class RENDER_MODEL>
		RENDER_MODEL* GetModel() const { return static_cast<RENDER_MODEL*>(m_models[RENDER_MODEL::id].get()); }
		const RenderModelArray& GetModels() const { return m_models; }

	protected:
		RenderModelArray m_models;
	};
}