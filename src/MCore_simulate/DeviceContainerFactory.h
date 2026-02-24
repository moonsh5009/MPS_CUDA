#pragma once

#include "../MCore_interface/IDeviceContainer.h"

#include <functional>

#include "HeaderPre.h"

namespace mcore::simulate
{
	class __MY_EXT_CLASS__ DeviceContainerFactory
	{
	public:
		using Func = std::function<std::shared_ptr<IDeviceContainer>(ISimulateManager*)>;

		static DeviceContainerFactory& Instance();

		bool Registry(DBTypeID id, Func&& func);
		DeviceContainerArray Build(ISimulateManager* pSimulateManager);

	private:
		std::array<Func, TYPE_ID_MAX> m_funcs;
	};
}

#include "HeaderPost.h"
