#pragma once

#include "DBDef.h"
#include "DBNotifyInfo.h"

#include <memory>
#include <type_traits>
#include <concepts>

namespace mcore
{
	struct IDBData;
	class IDBPool;
	class IDBTransactionCommand
	{
	public:
		IDBTransactionCommand() = delete;
		IDBTransactionCommand(
			const std::weak_ptr<IDBPool>& pDBPool,
			DBKey key,
			DBTypeID typeId,
			DBUpdateType undoType,
			DBUpdateType redoType,
			std::shared_ptr<IDBData>&& pUndoData,
			std::shared_ptr<IDBData>&& pRedoData)
			: m_pDBPool{ pDBPool }
			, m_key{ key }
			, m_typeId{ typeId }
			, m_undoType{ undoType }
			, m_redoType{ redoType }
			, m_pUndoData{ std::move(pUndoData) }
			, m_pRedoData{ std::move(pRedoData) }
		{}
		virtual ~IDBTransactionCommand() = default;
		IDBTransactionCommand(const IDBTransactionCommand&) = default;
		IDBTransactionCommand(IDBTransactionCommand&&) = default;
		IDBTransactionCommand& operator=(const IDBTransactionCommand&) = default;
		IDBTransactionCommand& operator=(IDBTransactionCommand&&) = default;

		virtual void Undo() const = 0;
		virtual void Redo() const = 0;

		constexpr DBKey GetKey() const { return m_key; }
		constexpr DBTypeID GetTypeID() const { return m_typeId; }
		constexpr DBUpdateType GetUndoType() const { return m_undoType; }
		constexpr DBUpdateType GetRedoType() const { return m_redoType; }
		constexpr const std::shared_ptr<IDBData>& GetUndoData() const { return m_pUndoData; }
		constexpr const std::shared_ptr<IDBData>& GetRedoData() const { return m_pRedoData; }

	protected:
		std::weak_ptr<IDBPool> m_pDBPool;
		DBKey m_key;
		DBTypeID m_typeId;
		DBUpdateType m_undoType;
		DBUpdateType m_redoType;
		std::shared_ptr<IDBData> m_pUndoData;
		std::shared_ptr<IDBData> m_pRedoData;
	};

	using DBTransactionCommandBuffer = std::vector<std::unique_ptr<IDBTransactionCommand>>;

	template <typename T>
	concept DerivedDBTransactionCommand = std::is_base_of_v<IDBTransactionCommand, T>;
}