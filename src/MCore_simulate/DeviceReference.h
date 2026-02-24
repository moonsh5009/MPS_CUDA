#pragma once

#include "../MCore_interface/ISimulateManager.h"
#include "../MCore_database/Ref.h"

#include "DeviceContainer.h"

namespace mcore::simulate
{
	template<DerivedDeviceContainer DEVICE_CONTAINER>
	class DeviceReference : public IDeviceContainer
	{
	public:
		using DB = DEVICE_CONTAINER::DB;
		using CONTAINER = DEVICE_CONTAINER;

		DeviceReference(ISimulateManager* pSimulateManager)
			: IDeviceContainer{ pSimulateManager }
		{}
		~DeviceReference() override = default;
		DeviceReference(const DeviceReference&) = default;
		DeviceReference(DeviceReference&&) = default;
		DeviceReference& operator=(const DeviceReference&) = default;
		DeviceReference& operator=(DeviceReference&&) = default;

		operator const CONTAINER& () const { return *GetSimulateManager()->GetDeviceContainer<CONTAINER>(); }
		const CONTAINER& operator*() const { return *GetSimulateManager()->GetDeviceContainer<CONTAINER>(); }
		const CONTAINER* operator->() const { return GetSimulateManager()->GetDeviceContainer<CONTAINER>().get(); }

		void Initialize() override {}

		DBTypeID GetTypeID() const final { return DB::META_DATA::id; }
		bool IsEmpty() const final
		{
			return GetSimulateManager()->GetDeviceContainer<CONTAINER>()->IsEmpty();
		}

		const mcuda::VKDeviceSingleArray<DBKey>& GetKeys() const
		{
			return reinterpret_cast<DeviceContainer<typename CONTAINER::DB>*>(GetSimulateManager()->GetDeviceContainer<CONTAINER>().get())->GetKeys();
		}

		void AddDB(const IDBData* newDB) final
		{}
		void ModifyDB(const IDBData* prevDB, const IDBData* newDB) final
		{}
		void DeleteDB(DBKey key, const IDBData* prevDB) final
		{}

		const CONTAINER& GetContainer() const
		{
			return *GetSimulateManager()->GetDeviceContainer<CONTAINER>();
		}
	};
}
