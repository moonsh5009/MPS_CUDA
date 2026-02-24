#pragma once

#include "../MCore_util/Signal.h"

#include "IDBPool.h"
#include "IDBTransaction.h"

#include <functional>

namespace mcore
{
	class IDBTransactionCommand;
	class IDBSession
	{
	public:
		IDBSession() = default;
		virtual ~IDBSession() = default;
		IDBSession(const IDBSession&) = delete;
		IDBSession(IDBSession&&) = default;
		IDBSession& operator=(const IDBSession&) = delete;
		IDBSession& operator=(IDBSession&&) = default;

		virtual void Initialize() = 0;
		virtual bool TransactionGuard(const std::function<bool(IDBSession*)>& func) = 0;

		virtual void Undo() const = 0;
		virtual void Redo() const = 0;

		template<DerivedDBTransactionCommand COMMAND, typename... Args>
		void PushCommand(Args&&... args)
		{
			m_pTransaction->PushCommand<COMMAND>(std::forward<Args>(args)...);
		}

		template<class DB_POOL>
			requires std::derived_from<DB_POOL, IDBPool>
		auto GetCastPool() const
		{
			return std::static_pointer_cast<DB_POOL>(m_aPool[DB_POOL::id]);
		}
		template<DerivedDBData DATA>
		const auto& GetPool() const
		{
			return m_aPool[DATA::META_DATA::id];
		}
		template<DBTypeID TYPE_ID>
		const auto& GetPool() const
		{
			return m_aPool[TYPE_ID];
		}
		std::shared_ptr<IDBPool> GetPool(DBTypeID typeId) const
		{
			if (typeId >= m_aPool.size())
				return nullptr;
			return m_aPool[typeId];
		}

		IDBTransaction* GetTransaction() const { return m_pTransaction.get(); }
		const std::vector<std::shared_ptr<IDBPool>>& GetAllPools() const { return m_aPool; }

		mcore::Signal<void(DBNotifyInfoArray)> commitSignal = mcore::MakeSignal<void(DBNotifyInfoArray)>();

	protected:
		std::unique_ptr<IDBTransaction> m_pTransaction;
		std::vector<std::shared_ptr<IDBPool>> m_aPool;
	};
}