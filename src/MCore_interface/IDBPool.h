#pragma once

#include "IDBData.h"

#include <memory>
#include <shared_mutex>

namespace mcore
{
	class IDBSession;
	class IDBPool : public std::enable_shared_from_this<IDBPool>
	{
	public:
		IDBPool() = delete;
		IDBPool(IDBSession* pDBSession) :
			m_pDBSession{ pDBSession }
		{}
		virtual ~IDBPool() = default;
		IDBPool(const IDBPool&) = delete;
		IDBPool(IDBPool&&) = default;
		IDBPool& operator=(const IDBPool&) = delete;
		IDBPool& operator=(IDBPool&&) = default;

		virtual void Initialize(size_t hashSize) = 0;

		virtual std::shared_ptr<const IDBData> GetBase(DBKey key) const = 0;
		
		virtual DBKey Insert(std::shared_ptr<IDBData>&& data) = 0;
		virtual void Set(std::shared_ptr<IDBData>&& data) = 0;
		virtual void Delete(DBKey key) = 0;

		virtual DBKey DirectInsert(std::shared_ptr<IDBData>&& data) = 0;
		virtual void DirectSet(std::shared_ptr<IDBData>&& data) = 0;
		virtual void DirectDelete(DBKey key) = 0;

		virtual size_t GetSize() const = 0;
		virtual bool IsEmpty() const { return GetSize() > 0; }

		IDBSession* GetDBSession() const { return m_pDBSession; }

	protected:
		IDBSession* m_pDBSession;
	};
}