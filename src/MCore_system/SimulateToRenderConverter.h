#pragma once

#include "../MCore_interface/ISimulateManager.h"
#include "../MCore_interface/IRenderModelContainer.h"

#include "SimulateToRenderConverterFactory.h"

namespace mcore::system
{
	template<DerivedDeviceContainer _DEVICE_CONTAINER>
	class SimulateToRenderConverter : public ISimulateToRenderConverter
	{
	public:
		using SIMULATE_CONTAINER = _DEVICE_CONTAINER;

		DBTypeID GetTypeID() const final
		{
			return SIMULATE_CONTAINER::DB::META_DATA::id;
		}

		void Convert(ISimulateManager* pSimulateManager, IRenderModelContainer* pRenderModelManager)
		{
			const auto& pDeviceContainer = pSimulateManager->GetDeviceContainer<SIMULATE_CONTAINER>();
			OnConvert(pDeviceContainer.get(), pRenderModelManager);
		}

		virtual void OnConvert(SIMULATE_CONTAINER* pSimulateContainer, IRenderModelContainer* pRenderModelManager) = 0;
	};
}

#define REGISTRY_SIMULATE_TO_RENDER(SIMULATE_TO_RENDER) \
	namespace { \
		const bool registered_##SIMULATE_TO_RENDER = mcore::system::SimulateToRenderConverterFactory::Instance().Registry(SIMULATE_TO_RENDER::SIMULATE_CONTAINER::DB::META_DATA::id, []() -> std::unique_ptr<mcore::ISimulateToRenderConverter> \
		{ \
			return std::make_unique<SIMULATE_TO_RENDER>(); \
		}); \
	}
