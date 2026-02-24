#pragma once

#include "../MCore_interface/IDBSession.h"
#include "Transaction.h"

#include "HeaderPre.h"

namespace mcore::database
{
	class __MY_EXT_CLASS__ Session final : public IDBSession
	{
	public:
		Session();

		void Initialize() final;
		bool TransactionGuard(const std::function<bool(IDBSession*)>& func) final;

		void Undo() const final;
		void Redo() const final;
	};
}

#include "HeaderPost.h"