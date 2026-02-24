#include "stdafx.h"
#include "Transaction.h"

#include <ranges>

using namespace mcore::database;

Transaction::Transaction(IDBSession* pDBSession) :
	IDBTransaction{ pDBSession }
{
	m_undo.reserve(100);
	m_redo.reserve(100);
}

void Transaction::BeginTrannsaction()
{
	SetState(DBTransactionState::RECORDING);
}

void Transaction::Commit()
{
	DispatchCommit(m_commands);
	m_undo.emplace_back(std::move(m_commands));
	m_redo.clear();

	SetState(DBTransactionState::IDLE);
}

void Transaction::RollBack()
{
	SetState(DBTransactionState::UNDOING);

	Undo(m_commands, false);
	m_commands.clear();

	SetState(DBTransactionState::IDLE);
}

void Transaction::Undo()
{
	if (m_undo.empty()) return;

	SetState(DBTransactionState::UNDOING);

	auto commands = std::move(m_undo.back());
	m_undo.pop_back();

	Undo(commands, true);
	m_redo.emplace_back(std::move(commands));

	SetState(DBTransactionState::IDLE);
}

void Transaction::Redo()
{
	SetState(DBTransactionState::REDOING);

	auto commands = std::move(m_redo.back());
	m_redo.pop_back();

	Redo(commands, true);
	m_undo.emplace_back(std::move(commands));

	SetState(DBTransactionState::IDLE);
}

void Transaction::Undo(const DBTransactionCommandBuffer& commands, bool bDispatch) const
{
	for (const auto& command : commands)
	{
		command->Undo();
	}

	if (bDispatch)
		DispatchUndo(commands);
}

void Transaction::Redo(const DBTransactionCommandBuffer& commands, bool bDispatch) const
{
	for (const auto& command : commands)
	{
		command->Redo();
	}
	if (bDispatch)
		DispatchRedo(commands);
}

void Transaction::DispatchCommit(const DBTransactionCommandBuffer& commands) const
{
	DBNotifyInfoArray notifyArray;
	notifyArray.reserve(commands.size());
	std::ranges::transform(commands, std::back_inserter(notifyArray), [](const auto& command)
	{
		return DBNotifyInfo
		{
			DBNotifyType::COMMIT,
			command->GetKey(),
			command->GetTypeID(),
			command->GetRedoType(),
			command->GetRedoData(),
			command->GetUndoData()
		};
	});

	GetDBSession()->commitSignal->Dispatch(notifyArray);
}

void Transaction::DispatchUndo(const DBTransactionCommandBuffer& commands) const
{
	DBNotifyInfoArray notifyArray;
	notifyArray.reserve(commands.size());
	std::ranges::transform(commands | std::views::reverse, std::back_inserter(notifyArray), [](const auto& command)
	{
		return DBNotifyInfo
		{
			DBNotifyType::UNDO,
			command->GetKey(),
			command->GetTypeID(),
			command->GetUndoType(),
			command->GetUndoData(),
			command->GetRedoData(),
		};
	});

	GetDBSession()->commitSignal->Dispatch(notifyArray);
}

void Transaction::DispatchRedo(const DBTransactionCommandBuffer& commands) const
{
	DBNotifyInfoArray notifyArray;
	notifyArray.reserve(commands.size());
	std::ranges::transform(commands, std::back_inserter(notifyArray), [](const auto& command)
	{
		return DBNotifyInfo
		{
			DBNotifyType::REDO,
			command->GetKey(),
			command->GetTypeID(),
			command->GetRedoType(),
			command->GetRedoData(),
			command->GetUndoData()
		};
	});

	GetDBSession()->commitSignal->Dispatch(notifyArray);
}