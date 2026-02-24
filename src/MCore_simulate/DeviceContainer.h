#pragma once

#include "../MCore_interface/IDBData.h"
#include "../MCore_util/HostSingleArray.h"

#include "DeviceContainerFactory.h"

namespace mcore::simulate
{
	struct DeviceObjectData
	{
		DBKey* keys;
		size_t count;
	};

	template<DerivedDBData DBDATA>
	class DeviceContainer : public IDeviceContainer
	{
	public:
		using DB = DBDATA;
		using DeviceData = DeviceObjectData;

		DeviceContainer(ISimulateManager* pSimulateManager)
			: IDeviceContainer{ pSimulateManager }
			, m_keys{ vk::BufferUsageFlagBits::eStorageBuffer }
		{}
		~DeviceContainer() override = default;
		DeviceContainer(const DeviceContainer&) = default;
		DeviceContainer(DeviceContainer&&) = default;
		DeviceContainer& operator=(const DeviceContainer&) = default;
		DeviceContainer& operator=(DeviceContainer&&) = default;

		void Initialize() override {}

		mcore::DBTypeID GetTypeID() const override { return DB::META_DATA::id; }
		bool IsEmpty() const final { return m_keys.IsEmpty(); }

		const mcuda::VKDeviceSingleArray<DBKey>& GetKeys() const { return m_keys; }
		mcuda::VKDeviceSingleArray<DBKey>& GetKeys() { return m_keys; }

		const std::vector<DBKey>& GetHostKeys() const { return m_hostKeys.GetBuffer(); }

		void AddDB(const IDBData* newDB) final
		{
			OnAddDB(reinterpret_cast<const DB*>(newDB));
			m_keyIndexMap.emplace(newDB->GetKey(), static_cast<IndexType>(m_keys.Insert(newDB->GetKey())));
			m_hostKeys.Insert(newDB->GetKey());
		}
		void ModifyDB(const IDBData* prevDB, const IDBData* newDB) final
		{
			const auto itrKeyIndex = m_keyIndexMap.find(newDB->GetKey());
			if (itrKeyIndex == m_keyIndexMap.end())
				throw std::runtime_error("Cannot modify non-existing DB data.");

			OnModifyDB(itrKeyIndex->second, reinterpret_cast<const DB*>(prevDB), reinterpret_cast<const DB*>(newDB));
		}
		void DeleteDB(DBKey key, const IDBData* prevDB) final
		{
			const auto itrKeyIndex = m_keyIndexMap.find(key);
			if (itrKeyIndex == m_keyIndexMap.end())
				throw std::runtime_error("Cannot modify non-existing DB data.");

			OnDeleteDB(itrKeyIndex->second, reinterpret_cast<const DB*>(prevDB));
			m_keys.Remove(itrKeyIndex->second);
			m_hostKeys.Remove(itrKeyIndex->second);
			m_keyIndexMap.erase(itrKeyIndex);
		}

		virtual void OnAddDB(const DB* newDB) {}
		virtual void OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB) {}
		virtual void OnDeleteDB(IndexType index, const DB* prevDB) {}

		IndexType GetIndex(DBKey key) const
		{
			const auto itrKeyIndex = m_keyIndexMap.find(key);
			if (itrKeyIndex == m_keyIndexMap.end())
				throw std::runtime_error("DB key not found in DeviceContainer.");
			return itrKeyIndex->second;
		}

		DeviceData GetDeviceData() const
		{
			DeviceData data;
			data.keys = m_keys.GetBuffer().GetData();
			data.count = m_keys.GetSize();
			return data;
		}

	protected:
		std::unordered_map<DBKey, IndexType> m_keyIndexMap;
		mcuda::VKDeviceSingleArray<DBKey> m_keys;
		mcuda::HostSingleArray<DBKey> m_hostKeys;
	};
}

#define REGISTRY_DEVICE_CONTAINER(DEVICE_CONTAINER) \
	namespace { \
		const bool registered_##DEVICE_CONTAINER = mcore::simulate::DeviceContainerFactory::Instance().Registry(DEVICE_CONTAINER::DB::META_DATA::id, [](mcore::ISimulateManager* pSimulateManager) -> std::shared_ptr<mcore::IDeviceContainer> \
		{ \
			auto pDeviceContainer = std::make_shared<DEVICE_CONTAINER>(pSimulateManager); \
			pDeviceContainer->Initialize(); \
			return pDeviceContainer; \
		}); \
	}
