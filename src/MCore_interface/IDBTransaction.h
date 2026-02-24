#pragma once

#include "IDBTransactionCommand.h"

namespace mcore
{
	enum class DBTransactionState
	{
		IDLE,
		RECORDING,
		UNDOING,
		REDOING,
	};

	class IDBSession;
	class IDBTransaction
	{
	public:
		IDBTransaction(IDBSession* pDBSession)
			: m_pDBSession{ pDBSession }
			, m_state{ DBTransactionState::IDLE }
		{}
		virtual ~IDBTransaction() = default;
		IDBTransaction(const IDBTransaction&) = delete;
		IDBTransaction(IDBTransaction&&) = default;
		IDBTransaction& operator=(const IDBTransaction&) = delete;
		IDBTransaction& operator=(IDBTransaction&&) = default;

		virtual void BeginTrannsaction() = 0;
		virtual void Commit() = 0;
		virtual void RollBack() = 0;

		virtual void Undo() = 0;
		virtual void Redo() = 0;

		template<DerivedDBTransactionCommand COMMAND, typename... Args>
		void PushCommand(Args&&... args)
		{
			m_commands.emplace_back(std::make_unique<COMMAND>(std::forward<Args>(args)...));
		}

		DBTransactionState GetState() const { return m_state; }
		bool IsActive() const { return GetState() == DBTransactionState::RECORDING; }

		IDBSession* GetDBSession() const { return m_pDBSession; }

	protected:
		void SetState(DBTransactionState state) { m_state = state; }

		IDBSession* m_pDBSession;

		DBTransactionState m_state;
		DBTransactionCommandBuffer m_commands;
	};
}