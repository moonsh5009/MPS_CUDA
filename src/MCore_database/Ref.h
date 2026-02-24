#pragma once

#include "../MCore_interface/IDBSession.h"
#include "../MCore_interface/DBAdapter.h"

namespace mcore::database
{
	template<DerivedDBData TYPE>
	struct Ref : public IDBData
	{
		Ref() = default;
		~Ref() override = default;
		Ref(const Ref&) = default;
		Ref(Ref&&) = default;
		Ref& operator=(const Ref&) = default;
		Ref& operator=(Ref&&) = default;

		operator const TYPE& () const { return *m_pData; }
		const TYPE& operator*() const { return *m_pData; }
		const TYPE* operator->() const { return m_pData.get(); }

		template<typename T = TYPE>
		decltype(auto) operator[](size_t i) const requires requires(const T& t, size_t idx) { t[idx]; }
		{
			return (*m_pData)[i];
		}

		void Initialize(const IDBSession* pDBSession) final
		{
			m_pData.reset();
			m_pPool = pDBSession->GetPool<TYPE>();
		}

		DBKey GetKey() const final
		{
			return m_pData ? m_pData->GetKey() : NullKey;
		}

		void SetKey(DBKey key) final
		{
			if (const auto pPool = m_pPool.lock())
			{
				if (const auto pData = pPool->GetBase(key))
				{
					m_pData = std::static_pointer_cast<const TYPE>(std::const_pointer_cast<const IDBData>(pData));
					return;
				}
			}
			m_pData.reset();
		}

		std::shared_ptr<IDBData> Clone() const final
		{
			return std::make_shared<Ref<TYPE>>(*this);
		}

		DBTypeID GetTypeID() const final
		{
			return TYPE::META_DATA::id;
		}

		size_t GetStreamSize() const final
		{
			return sizeof(DBKey);
		}

		std::byte* ToStream(std::byte* pStream) const final
		{
			const auto streamSize = GetStreamSize();
			const auto key = GetKey();
			std::memcpy(pStream, &key, streamSize);
			return pStream + streamSize;
		}

		const std::byte* FromStream(const std::byte* pStream) final
		{
			const auto streamSize = GetStreamSize();
			DBKey key = NullKey;
			std::memcpy(&key, pStream, streamSize);
			SetKey(key);
			return pStream + streamSize;
		}

	private:
		std::weak_ptr<IDBPool> m_pPool;
		std::shared_ptr<const TYPE> m_pData;
	};
}