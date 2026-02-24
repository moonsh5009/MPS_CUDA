#pragma once

#include "RenderTargetFactory.h"

#define DECLARE_RENDER_TARGET(RENDER_TARGET) \
	public: \
		static size_t id; \
	private:

#define IMPLEMENT_RENDER_TARGET(RENDER_TARGET, TYPE) \
	size_t RENDER_TARGET::id = static_cast<size_t>(TYPE); \
	namespace \
	{ \
		const auto registry_##RENDER_TARGET = mcore::render::RenderTargetFactory::Instance().Registry(TYPE, [](auto pScene) \
		{ \
			return std::make_unique<RENDER_TARGET>(pScene); \
		}); \
	}