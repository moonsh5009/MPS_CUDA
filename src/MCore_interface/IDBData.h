#pragma once

#include "DBDef.h"
#include "DBAdapter.h"

namespace mcore
{
	class IDBSession;
	struct IDBData
	{
		IDBData() = default;
		virtual ~IDBData() = default;
		IDBData(const IDBData&) = default;
		IDBData(IDBData&&) = default;
		IDBData& operator=(const IDBData&) = default;
		IDBData& operator=(IDBData&&) = default;

		virtual void Initialize(const IDBSession* pDBSession) {}

		virtual DBKey GetKey() const = 0;
		virtual void SetKey(DBKey key) = 0;

		virtual std::shared_ptr<IDBData> Clone() const = 0;

		virtual DBTypeID GetTypeID() const = 0;

		virtual size_t GetStreamSize() const { return 0; }
		virtual std::byte* ToStream(std::byte* pStream) const { return pStream; }
		virtual const std::byte* FromStream(const std::byte* pStream) { return pStream; }
	};

	template <typename T>
	concept DerivedDBData = std::derived_from<T, IDBData>;
}

template<mcore::DerivedDBData T>
struct DBAdapter<T>
{
	static size_t GetStreamSize(const T& data)
	{
		return data.GetStreamSize();
	}
	static std::byte* ToStream(std::byte* pStream, const T& data)
	{
		return data.ToStream(pStream);
	}
	static const std::byte* FromStream(const std::byte* pStream, T& data)
	{
		return data.FromStream(pStream);
	}
};