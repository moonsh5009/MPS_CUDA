#pragma once

#include "../MCore_util/DeviceSingleArray.h"

#include "DBDef.h"

#include <memory>

namespace mcore
{
	struct IDBData;
	class ISimulateManager;
	class IDeviceContainer
	{
	public:
		IDeviceContainer() = default;
		IDeviceContainer(ISimulateManager* pSimulateManager)
			: m_pSimulateManager{ pSimulateManager }
		{}
		virtual ~IDeviceContainer() = default;
		IDeviceContainer(const IDeviceContainer&) = default;
		IDeviceContainer(IDeviceContainer&&) = default;
		IDeviceContainer& operator=(const IDeviceContainer&) = default;
		IDeviceContainer& operator=(IDeviceContainer&&) = default;

		virtual void Initialize() = 0;

		virtual DBTypeID GetTypeID() const = 0;
		virtual bool IsEmpty() const = 0;

		virtual void AddDB(const IDBData* newDB) = 0;
		virtual void ModifyDB(const IDBData* prevDB, const IDBData* newDB) = 0;
		virtual void DeleteDB(DBKey key, const IDBData* prevDB) = 0;

		ISimulateManager* GetSimulateManager() const { return m_pSimulateManager; }

	protected:
		ISimulateManager* m_pSimulateManager;
	};

	template <typename T>
	concept DerivedDeviceContainer = std::derived_from<T, IDeviceContainer>;

	using DeviceContainerArray = std::vector<std::shared_ptr<IDeviceContainer>>;
}
