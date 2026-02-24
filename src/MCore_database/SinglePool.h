#pragma once

#include "KeyGenerator.h"
#include "SinglePoolBase.h"
#include "PoolFactory.h"

#include "HeaderPre.h"

namespace mcore::database
{
	template<DerivedDBData DATA>
	class SinglePool : public SinglePoolBase
	{
	public:
		SinglePool(IDBSession* pDBSession)
			: SinglePoolBase{ std::move(pDBSession), std::make_shared<DATA>() }
		{}

		std::shared_ptr<DATA> CloneData() const
		{
			return std::static_pointer_cast<DATA>(Get()->Clone());
		}

		std::shared_ptr<const DATA> Get() const
		{
			return std::static_pointer_cast<const DATA>(GetBase(1));
		}
		void Set(std::shared_ptr<DATA>&& data)
		{
			SinglePoolBase::Set(std::static_pointer_cast<IDBData>(std::move(data)));
		}

	private:
		using SinglePoolBase::Insert;
		using SinglePoolBase::Set;
		using SinglePoolBase::DirectInsert;
		using SinglePoolBase::DirectSet;
	};
}

#define DECLARE_SINGLE_DATAPOOL(DPOOL, NAME) \
	public: \
		static constexpr auto id = DBMetaData##NAME::id; \
	private:

#define IMPLEMENT_SINGLE_DATAPOOL(DPOOL, NAME) \
namespace \
{ \
	const auto registry_##DPOOL = mcore::database::PoolFactory::Instance().Registry(DPOOL::id, [](mcore::IDBSession* pDBSession) { \
		auto pPool = std::make_shared<DPOOL>(pDBSession); \
		pPool->Initialize(1); \
		return pPool; \
	}); \
}

#include "HeaderPost.h"