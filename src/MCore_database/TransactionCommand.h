#pragma once

#include "../MCore_interface/IDBSession.h"
#include "../MCore_interface/IDBTransactionCommand.h"

#include <memory>
#include <vector>
#include <functional>

namespace mcore::database
{
	class TransactionSetCommand : public IDBTransactionCommand
	{
	public:
		TransactionSetCommand() = delete;
		TransactionSetCommand(
			const std::weak_ptr<IDBPool>& pDBPool,
			std::shared_ptr<IDBData>&& pUndoData,
			std::shared_ptr<IDBData>&& pRedoData)
			: IDBTransactionCommand{
				pDBPool,
				pUndoData->GetKey(),
				pUndoData->GetTypeID(),
				DBUpdateType::MODIFY,
				DBUpdateType::MODIFY,
				std::move(pUndoData),
				std::move(pRedoData)
			}
		{}
		virtual ~TransactionSetCommand() = default;
		TransactionSetCommand(const TransactionSetCommand&) = delete;
		TransactionSetCommand(TransactionSetCommand&&) = default;
		TransactionSetCommand& operator=(const TransactionSetCommand&) = delete;
		TransactionSetCommand& operator=(TransactionSetCommand&&) = default;

		void Undo() const override;
		void Redo() const override;
	};

	class TransactionInsertCommand : public IDBTransactionCommand
	{
	public:
		TransactionInsertCommand() = delete;
		TransactionInsertCommand(
			const std::weak_ptr<IDBPool>& pDBPool,
			std::shared_ptr<IDBData>&& pRedoData)
			: IDBTransactionCommand{
				pDBPool,
				pRedoData->GetKey(),
				pRedoData->GetTypeID(),
				DBUpdateType::REMOVE,
				DBUpdateType::ADD,
				{},
				std::move(pRedoData)
			}
		{}
		virtual ~TransactionInsertCommand() = default;
		TransactionInsertCommand(const TransactionInsertCommand&) = delete;
		TransactionInsertCommand(TransactionInsertCommand&&) = default;
		TransactionInsertCommand& operator=(const TransactionInsertCommand&) = delete;
		TransactionInsertCommand& operator=(TransactionInsertCommand&&) = default;

		void Undo() const override;
		void Redo() const override;
	};

	class TransactionDeleteCommand : public IDBTransactionCommand
	{
	public:
		TransactionDeleteCommand() = delete;
		TransactionDeleteCommand(
			const std::weak_ptr<IDBPool>& pDBPool,
			std::shared_ptr<IDBData>&& pUndoData)
			: IDBTransactionCommand{
				pDBPool,
				pUndoData->GetKey(),
				pUndoData->GetTypeID(),
				DBUpdateType::ADD,
				DBUpdateType::REMOVE,
				std::move(pUndoData),
				{}
			}
		{}
		virtual ~TransactionDeleteCommand() = default;
		TransactionDeleteCommand(const TransactionDeleteCommand&) = delete;
		TransactionDeleteCommand(TransactionDeleteCommand&&) = default;
		TransactionDeleteCommand& operator=(const TransactionDeleteCommand&) = delete;
		TransactionDeleteCommand& operator=(TransactionDeleteCommand&&) = default;

		void Undo() const override;
		void Redo() const override;
	};
}