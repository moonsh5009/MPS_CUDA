#pragma once

#include "../MCore_util/Signal.h"
#include "../MCore_interface/IDBSession.h"
#include "../MCore_interface/IDBTransaction.h"

#include "TransactionCommand.h"
#include <shared_mutex>

#include "HeaderPre.h"

namespace mcore::database
{
	class __MY_EXT_CLASS__ Transaction : public IDBTransaction
	{
	public:
		Transaction(IDBSession* pDBSession);
		~Transaction() = default;
		Transaction(const Transaction&) = delete;
		Transaction(Transaction&&) = default;
		Transaction& operator=(const Transaction&) = delete;
		Transaction& operator=(Transaction&&) = default;

		void BeginTrannsaction() override;
		void Commit() override;
		void RollBack() override;

		void Undo() override;
		void Redo() override;

	private:
		void Undo(const DBTransactionCommandBuffer& commands, bool bDispatch) const;
		void Redo(const DBTransactionCommandBuffer& commands, bool bDispatch) const;

		void DispatchCommit(const DBTransactionCommandBuffer& commands) const;
		void DispatchUndo(const DBTransactionCommandBuffer& commands) const;
		void DispatchRedo(const DBTransactionCommandBuffer& commands) const;

		std::vector<DBTransactionCommandBuffer> m_undo;
		std::vector<DBTransactionCommandBuffer> m_redo;
	};
}

#include "HeaderPost.h"