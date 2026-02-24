#pragma once

#include "RenderPassFactory.h"

#define DECLARE_RENDER_PASS(RENDER_PASS) \
	public: \
		static size_t id; \
	private:

#define IMPLEMENT_RENDER_PASS(RENDER_PASS, TYPE) \
	size_t RENDER_PASS::id = static_cast<size_t>(TYPE); \
	namespace \
	{ \
		const auto registry_##RENDER_PASS = mcore::render::RenderPassFactory::Instance().Registry(TYPE, [](auto pRenderingEngine) \
		{ \
			return std::make_shared<RENDER_PASS>(pRenderingEngine); \
		}); \
	}