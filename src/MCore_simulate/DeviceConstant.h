#pragma once

#include "../MCore_interface/IDBData.h"

#include "DeviceContainerFactory.h"

namespace mcore::simulate
{
	template<DerivedDBData DBDATA>
	class DeviceConstant : public IDeviceContainer
	{
	public:
		using DB = DBDATA;

		DeviceConstant(ISimulateManager* pSimulateManager)
			: IDeviceContainer{ pSimulateManager }
		{}
		~DeviceConstant() override = default;
		DeviceConstant(const DeviceConstant&) = default;
		DeviceConstant(DeviceConstant&&) = default;
		DeviceConstant& operator=(const DeviceConstant&) = default;
		DeviceConstant& operator=(DeviceConstant&&) = default;

		void Initialize() override {}

		mcore::DBTypeID GetTypeID() const override { return DB::META_DATA::id; }
		bool IsEmpty() const final { return false; }

		void AddDB(const IDBData* newDB) final {}
		void ModifyDB(const IDBData* prevDB, const IDBData* newDB) final
		{
			OnModifyDB(reinterpret_cast<const DB*>(prevDB), reinterpret_cast<const DB*>(newDB));
		}
		void DeleteDB(DBKey key, const IDBData* prevDB) final {}

		virtual void OnModifyDB(const DB* prevDB, const DB* newDB) {}
	};
}

#define REGISTRY_DEVICE_CONSTANT(DEVICE_CONSTANT) \
	namespace { \
		const bool registered_##DEVICE_CONSTANT = mcore::simulate::DeviceContainerFactory::Instance().Registry(DEVICE_CONSTANT::DB::META_DATA::id, [](mcore::ISimulateManager* pSimulateManager) -> std::shared_ptr<mcore::IDeviceContainer> \
		{ \
			auto pDeviceConstant = std::make_shared<DEVICE_CONSTANT>(pSimulateManager); \
			pDeviceConstant->Initialize(); \
			return pDeviceConstant; \
		}); \
	}
