#include "stdafx.h"
#include "Session.h"

#include "PoolFactory.h"

using namespace mcore;
using namespace mcore::database;

Session::Session()
{}

void Session::Initialize()
{
	m_aPool = PoolFactory::Instance().Build(this);
	m_pTransaction = std::make_unique<Transaction>(this);
}

bool Session::TransactionGuard(const std::function<bool(IDBSession*)>& func)
{
	m_pTransaction->BeginTrannsaction();
	const auto res = func(this);
	if (!res)
	{
		m_pTransaction->RollBack();
		return false;
	}
	m_pTransaction->Commit();
	return true;
}

void Session::Undo() const
{
	m_pTransaction->Undo();
}

void Session::Redo() const
{
	m_pTransaction->Redo();
}