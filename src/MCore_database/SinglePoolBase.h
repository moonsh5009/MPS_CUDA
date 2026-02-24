#pragma once

#include "../MCore_interface/IDBPool.h"

#include "HeaderPre.h"

namespace mcore::database
{
	class __MY_EXT_CLASS__ SinglePoolBase : public IDBPool
	{
	public:
		SinglePoolBase(IDBSession* pSession, std::shared_ptr<IDBData>&& pData);

		void Initialize(size_t hashSize) override;

		std::shared_ptr<const IDBData> GetBase(DBKey key) const final;

		DBKey Insert(std::shared_ptr<IDBData>&& data) final;
		void Set(std::shared_ptr<IDBData>&& data) final;
		void Delete(DBKey key) final;

		void DirectSet(std::shared_ptr<IDBData>&& data) final;
		DBKey DirectInsert(std::shared_ptr<IDBData>&& data) final;
		void DirectDelete(DBKey key) final;

		size_t GetSize() const final { return 1; }

	protected:
		std::shared_ptr<IDBData> m_pData;
	};
}

#include "HeaderPost.h"