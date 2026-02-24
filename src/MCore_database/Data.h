#pragma once

#include "../MCore_util/MacroUtil.h"

#include "Ref.h"

#define DATABASE_FIELD_DECLARE_IMPL(TYPE, VAR, NAME)			UNWRAP TYPE VAR;
#define DATABASE_FIELD_DECLARE(ARG, INDEX)						EXPAND_ARGS(DATABASE_FIELD_DECLARE_IMPL, UNWRAP ARG)

#define DATABASE_FIELD_GET_STREAM_SIZE_IMPL(TYPE, VAR, NAME)	size += DBAdapter<UNWRAP TYPE>::GetStreamSize(VAR);
#define DATABASE_FIELD_GET_STREAM_SIZE(ARG, INDEX)				EXPAND_ARGS(DATABASE_FIELD_GET_STREAM_SIZE_IMPL, UNWRAP ARG)

#define DATABASE_FIELD_TO_STREAM_IMPL(TYPE, VAR, NAME)			pStream = DBAdapter<UNWRAP TYPE>::ToStream(pStream, VAR);
#define DATABASE_FIELD_TO_STREAM(ARG, INDEX)					EXPAND_ARGS(DATABASE_FIELD_TO_STREAM_IMPL, UNWRAP ARG)

#define DATABASE_FIELD_FROM_STREAM_IMPL(TYPE, VAR, NAME)		pStream = DBAdapter<UNWRAP TYPE>::FromStream(pStream, VAR);
#define DATABASE_FIELD_FROM_STREAM(ARG, INDEX)					EXPAND_ARGS(DATABASE_FIELD_FROM_STREAM_IMPL, UNWRAP ARG)

#define DATABASE_FIELD_1(CLASS, BASE_CLASS, ARGS) \
    FOR_EACH(DATABASE_FIELD_DECLARE, UNWRAP ARGS) \
	std::shared_ptr<mcore::IDBData> Clone() const override \
	{ \
		return std::make_shared<CLASS>(*this); \
	} \
	size_t GetStreamSize() const override \
	{ \
		size_t size = BASE_CLASS::GetStreamSize(); \
        FOR_EACH(DATABASE_FIELD_GET_STREAM_SIZE, UNWRAP ARGS) \
		return size; \
	} \
	std::byte* ToStream(std::byte* pStream) const override \
	{ \
		pStream = BASE_CLASS::ToStream(pStream); \
        FOR_EACH(DATABASE_FIELD_TO_STREAM, UNWRAP ARGS) \
		return pStream; \
	} \
	const std::byte* FromStream(const std::byte* pStream) override \
	{ \
		pStream = BASE_CLASS::FromStream(pStream); \
        FOR_EACH(DATABASE_FIELD_FROM_STREAM, UNWRAP ARGS) \
		return pStream; \
	} \
	mcore::DBTypeID GetTypeID() const override { return 0; } \
	using Parent = BASE_CLASS

#define DATABASE_FIELD_2(CLASS, BASE_CLASS, ARGS, NAME) \
    FOR_EACH(DATABASE_FIELD_DECLARE, UNWRAP ARGS) \
	std::shared_ptr<mcore::IDBData> Clone() const override \
	{ \
		return std::make_shared<CLASS>(*this); \
	} \
	size_t GetStreamSize() const override \
	{ \
		size_t size = BASE_CLASS::GetStreamSize(); \
        FOR_EACH(DATABASE_FIELD_GET_STREAM_SIZE, UNWRAP ARGS) \
		return size; \
	} \
	std::byte* ToStream(std::byte* pStream) const override \
	{ \
		pStream = BASE_CLASS::ToStream(pStream); \
        FOR_EACH(DATABASE_FIELD_TO_STREAM, UNWRAP ARGS) \
		return pStream; \
	} \
	const std::byte* FromStream(const std::byte* pStream) override \
	{ \
		pStream = BASE_CLASS::FromStream(pStream); \
        FOR_EACH(DATABASE_FIELD_FROM_STREAM, UNWRAP ARGS) \
		return pStream; \
	} \
	mcore::DBTypeID GetTypeID() const override \
	{ \
		return DBMetaData##NAME::id; \
	} \
	using META_DATA = DBMetaData##NAME; \
	using Parent = BASE_CLASS

#define DATABASE_FIELD(...) \
	EXPAND(EXPAND_OVERLOAD_4(__VA_ARGS__, DATABASE_FIELD_2, DATABASE_FIELD_1)(__VA_ARGS__))

namespace mcore::database
{
	struct Data : public IDBData
	{
		DATABASE_FIELD(Data, IDBData, (
			((DBKey), key, "KEY")
		));

		Data() = default;
		~Data() override = default;
		Data(const Data&) = default;
		Data(Data&&) = default;
		Data& operator=(const Data&) = default;
		Data& operator=(Data&&) = default;

		DBKey GetKey() const final { return key; }
		void SetKey(DBKey key) final { this->key = key; }
		void Initialize(const IDBSession* pDBSession) override
		{
			IDBData::Initialize(pDBSession);
			key = NullKey;
		}
	};
}