#pragma once

#include "../MCore_interface/ISimulateToRenderConverter.h"

#include <functional>

#include "HeaderPre.h"

namespace mcore::system
{
	class __MY_EXT_CLASS__ SimulateToRenderConverterFactory
	{
	public:
		using Func = std::function<std::unique_ptr<ISimulateToRenderConverter>()>;

		static SimulateToRenderConverterFactory& Instance();

		bool Registry(DBTypeID id, Func&& func);
		SimulateToRenderConverterArray Build();

		const size_t& GetCount() const { return m_count; }

	private:
		size_t m_count = 0;
		std::array<Func, TYPE_ID_MAX> m_funcs;
	};
}

#include "HeaderPost.h"