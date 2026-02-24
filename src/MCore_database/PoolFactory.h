#pragma once

#include "../MCore_interface/DBDef.h"

#include <memory>
#include <functional>

#include "HeaderPre.h"

namespace mcore
{
	class IDBPool;
	class IDBSession;
}
namespace mcore::database
{
	class __MY_EXT_CLASS__ PoolFactory
	{
	public:
		using Func = std::function<std::shared_ptr<IDBPool>(IDBSession*)>;

		static PoolFactory& Instance();

		bool Registry(DBTypeID id, Func&& func);
		std::vector<std::shared_ptr<IDBPool>> Build(IDBSession* pDBSession);

	private:
		DBTypeID m_size = 0;
		std::unordered_map<DBTypeID, Func> m_funcs;
	};
}

#include "HeaderPost.h"