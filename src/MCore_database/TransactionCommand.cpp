#include "stdafx.h"
#include "TransactionCommand.h"

#include "Pool.h"
#include "Transaction.h"

using namespace mcore::database;

void TransactionSetCommand::Undo() const
{
	if (const auto pPool = m_pDBPool.lock())
	{
		pPool->DirectSet(m_pUndoData->Clone());
	}
}

void TransactionSetCommand::Redo() const
{
	if (const auto pPool = m_pDBPool.lock())
	{
		pPool->DirectSet(m_pRedoData->Clone());
	}
}

void TransactionInsertCommand::Undo() const
{
	if (const auto pPool = m_pDBPool.lock())
	{
		pPool->DirectDelete(GetKey());
	}
}

void TransactionInsertCommand::Redo() const
{
	if (const auto pPool = m_pDBPool.lock())
	{
		pPool->DirectInsert(m_pRedoData->Clone());
	}
}

void TransactionDeleteCommand::Undo() const
{
	if (const auto pPool = m_pDBPool.lock())
	{
		pPool->DirectInsert(m_pUndoData->Clone());
	}
}

void TransactionDeleteCommand::Redo() const
{
	if (const auto pPool = m_pDBPool.lock())
	{
		pPool->DirectDelete(GetKey());
	}
}