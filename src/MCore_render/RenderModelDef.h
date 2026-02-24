#pragma once

#include "RenderModelFactory.h"

#define DECLARE_RENDER_MODEL(RENDER_MODEL) \
	public: \
		static size_t id; \
		virtual mcore::render::RenderModelType GetType() const override { return mcore::from_sizet<mcore::render::RenderModelType>(id); } \
		virtual uint32_t GetBindGroupInstance() const override { return static_cast<uint32_t>(id); } \
	private:

#define IMPLEMENT_RENDER_MODEL(RENDER_MODEL, TYPE) \
	size_t RENDER_MODEL::id = static_cast<size_t>(TYPE); \
	namespace \
	{ \
		const auto registry_##RENDER_MODEL = mcore::render::RenderModelFactory::Instance().Registry(TYPE, [](auto pRenderModelContainer) \
		{ \
			return std::make_unique<RENDER_MODEL>(pRenderModelContainer); \
		}); \
	}