#include "stdafx.h"
#include "PoolBase.h"

#include "Session.h"

using namespace mcore;
using namespace mcore::database;

PoolBase::PoolBase(IDBSession* pDBSession)
	: IDBPool{ pDBSession }
{}

void PoolBase::Initialize(size_t hashSize)
{
	m_dataArray.resize(hashSize);
}

std::shared_ptr<const IDBData> PoolBase::GetBase(DBKey key) const
{
	if (key >= m_dataArray.size())
		return {};
	return std::const_pointer_cast<const IDBData>(m_dataArray[key]);
}

DBKey PoolBase::Insert(std::shared_ptr<IDBData>&& data)
{
	const auto key = [&]
	{
		if (data->GetKey() == 0)
			data->SetKey(m_keyGenerator.NewKey());
		else
			m_keyGenerator.InsertKey(data->GetKey());
		return data->GetKey();
	}();

	GetDBSession()->PushCommand<TransactionInsertCommand>(
		weak_from_this(),
		data->Clone());

	DirectInsert(std::move(data));
	return key;
}

void PoolBase::Set(std::shared_ptr<IDBData>&& data)
{
	auto originData = GetBase(data->GetKey());
	GetDBSession()->PushCommand<TransactionSetCommand>(
		weak_from_this(),
		originData->Clone(),
		data->Clone());

	DirectSet(std::move(data));
}

void PoolBase::Delete(DBKey key)
{
	auto originData = GetBase(key);
	GetDBSession()->PushCommand<TransactionInsertCommand>(
		weak_from_this(),
		originData->Clone());

	DirectDelete(key);
}

DBKey PoolBase::DirectInsert(std::shared_ptr<IDBData>&& data)
{
	const auto key = [&]
	{
		if (data->GetKey() == 0)
			data->SetKey(m_keyGenerator.NewKey());
		else
			m_keyGenerator.InsertKey(data->GetKey());
		return data->GetKey();
	}();

	if (key >= m_dataArray.size())
		return 0;
	if (m_dataArray[key])
		return 0;
	m_dataArray[key] = std::move(data);
	return key;
}

void PoolBase::DirectSet(std::shared_ptr<IDBData>&& data)
{
	const auto key = data->GetKey();
	if (key >= m_dataArray.size())
		return;
	if (!m_dataArray[key])
		return;
	m_dataArray[key] = std::move(data);
}

void PoolBase::DirectDelete(DBKey key)
{
	if (key >= m_dataArray.size())
		return;
	if (!m_dataArray[key])
		return;
	m_dataArray[key].reset();
	m_keyGenerator.DeleteKey(key);
}