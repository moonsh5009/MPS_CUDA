#include "stdafx.h"
#include "PoolFactory.h"

#include "../MCore_interface/IDBSession.h"

using namespace mcore;
using namespace mcore::database;

PoolFactory& PoolFactory::Instance()
{
	static PoolFactory singleton;
	return singleton;
}

bool PoolFactory::Registry(DBTypeID id, Func&& func)
{
	if (m_funcs.emplace(id, std::move(func)).second)
	{
		std::cerr << "Duplicated Database ID\n";
	}
	m_size = std::max(m_size, id + 1);
	return true;
}

std::vector<std::shared_ptr<IDBPool>> PoolFactory::Build(IDBSession* pDBSession)
{
	std::vector<std::shared_ptr<IDBPool>> aDBPool;
	aDBPool.resize(m_size);
	for (const auto& [id, func] : m_funcs)
	{
		aDBPool[id] = func(pDBSession);
	}
	return aDBPool;
}
