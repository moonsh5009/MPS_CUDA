#pragma once

#include "IDBData.h"
#include <memory>

namespace mcore
{
	enum class DBUpdateType : uint32_t
	{
		ADD = 0,
		MODIFY,
		REMOVE,
	};

	enum class DBNotifyType : uint32_t
	{
		COMMIT = 0,
		UNDO,
		REDO,
	};

	struct DBNotifyInfo
	{
		DBNotifyType type;
		DBKey key;
		DBTypeID typeId;
		DBUpdateType updateType;
		std::shared_ptr<IDBData> pData;
		std::shared_ptr<IDBData> pPrevData;
	};
	using DBNotifyInfoArray = std::vector<DBNotifyInfo>;
}