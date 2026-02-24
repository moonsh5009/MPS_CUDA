#include "stdafx.h"
#include "SinglePoolBase.h"

#include "Session.h"

using namespace mcore;
using namespace mcore::database;

SinglePoolBase::SinglePoolBase(IDBSession* pSession, std::shared_ptr<IDBData>&& pData)
	: IDBPool{ std::move(pSession) }
	, m_pData{ std::move(pData) }
{}

void SinglePoolBase::Initialize(size_t hashSize)
{
	m_pData->Initialize(GetDBSession());
	m_pData->SetKey(1);
}

std::shared_ptr<const IDBData> SinglePoolBase::GetBase(DBKey key) const
{
	return std::const_pointer_cast<const IDBData>(m_pData);
}

DBKey SinglePoolBase::Insert(std::shared_ptr<IDBData>&& data)
{
	assert(false);
	return 0;
}

void SinglePoolBase::Set(std::shared_ptr<IDBData>&& data)
{
	data->SetKey(1);
	GetDBSession()->PushCommand<TransactionSetCommand>(
		weak_from_this(),
		m_pData->Clone(),
		data->Clone());
	m_pData = std::move(data);
}

void SinglePoolBase::Delete(DBKey key)
{
	assert(false);
	return;
}

DBKey SinglePoolBase::DirectInsert(std::shared_ptr<IDBData>&& data)
{
	assert(false);
	return 0;
}

void SinglePoolBase::DirectSet(std::shared_ptr<IDBData>&& data)
{
	assert(m_pData);
	m_pData = std::move(data);
}

void SinglePoolBase::DirectDelete(DBKey key)
{
	assert(false);
	return;
}