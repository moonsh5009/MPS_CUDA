#pragma once

#include "PoolBase.h"
#include "PoolFactory.h"

#include "HeaderPre.h"

namespace mcore::database
{
	template<DerivedDBData DATA>
	class Pool : public PoolBase
	{
	public:
		Pool(IDBSession* pDBSession)
			: PoolBase{ std::move(pDBSession) }
		{}

		std::shared_ptr<DATA> NewData() const
		{
			auto pData = std::make_shared<DATA>();
			pData->Initialize(GetDBSession());
			return pData;
		}

		std::shared_ptr<DATA> CloneData(DBKey key) const
		{
			return std::static_pointer_cast<DATA>(GetBase(key)->Clone());
		}

		std::shared_ptr<const DATA> Get(DBKey key) const
		{
			return std::static_pointer_cast<const DATA>(GetBase(key));
		}
		DBKey Insert(std::shared_ptr<DATA>&& data)
		{
			return PoolBase::Insert(std::static_pointer_cast<IDBData>(std::move(data)));
		}
		void Set(std::shared_ptr<DATA>&& data)
		{
			PoolBase::Set(std::static_pointer_cast<IDBData>(std::move(data)));
		}

	private:
		using PoolBase::Insert;
		using PoolBase::Set;
		using PoolBase::DirectInsert;
		using PoolBase::DirectSet;
	};
}

#define DECLARE_DATAPOOL(DPOOL, NAME) \
	public: \
		static constexpr auto id = DBMetaData##NAME::id; \
		static constexpr auto hash_size = DBMetaData##NAME::hash_size; \
	private:

#define IMPLEMENT_DATAPOOL(DPOOL, NAME) \
namespace \
{ \
	const auto registry_##DPOOL = mcore::database::PoolFactory::Instance().Registry(DPOOL::id, [](mcore::IDBSession* pDBSession) { \
		auto pPool = std::make_shared<DPOOL>(pDBSession); \
		pPool->Initialize(DPOOL::hash_size); \
		return pPool; \
	}); \
}

#include "HeaderPost.h"