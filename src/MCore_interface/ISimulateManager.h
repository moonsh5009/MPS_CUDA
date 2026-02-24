#pragma once

#include "IDeviceContainer.h"
#include "ISimulateStep.h"

namespace mcore
{
	class ISimulateManager
	{
	public:
		ISimulateManager() = default;
		virtual ~ISimulateManager() = default;
		ISimulateManager(const ISimulateManager&) = delete;
		ISimulateManager(ISimulateManager&&) = default;
		ISimulateManager& operator=(const ISimulateManager&) = delete;
		ISimulateManager& operator=(ISimulateManager&&) = default;

		virtual void Initialize() = 0;

		virtual void Simulate() = 0;

		template<DerivedDeviceContainer DEVICE_CONTAINER>
		std::shared_ptr<DEVICE_CONTAINER> GetDeviceContainer() const
		{
			const auto typeId = DEVICE_CONTAINER::DB::META_DATA::id;
			return std::static_pointer_cast<DEVICE_CONTAINER>(GetDeviceContainer(typeId));
		}
		const std::shared_ptr<IDeviceContainer>& GetDeviceContainer(DBTypeID typeId) const
		{
			static const std::shared_ptr<IDeviceContainer> nullPtr = nullptr;
			if (typeId >= m_deviceContainers.size())
				return nullPtr;
			return m_deviceContainers[typeId];
		}
		const DeviceContainerArray& GetDeviceContainers() { return m_deviceContainers; }

		const SimulateStepGroupArray& GetStepGroups() const { return m_stepGroups; }

		template<typename T>
		T* FindStep() const
		{
			for (const auto& group : m_stepGroups)
				for (const auto& pStep : group)
					if (auto* p = dynamic_cast<T*>(pStep.get()))
						return p;
			return nullptr;
		}

	protected:
		DeviceContainerArray m_deviceContainers;
		SimulateStepGroupArray m_stepGroups;
	};
}
