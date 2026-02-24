#pragma once

#include "../MCore_interface/IDBPool.h"

#include "KeyGenerator.h"

#include "HeaderPre.h"

namespace mcore::database
{
	class __MY_EXT_CLASS__ PoolBase : public IDBPool
	{
	public:
		class Iterator
		{
		public:
			using iterator_category = std::forward_iterator_tag;
			using value_type = std::shared_ptr<IDBData>;
			using difference_type = std::ptrdiff_t;
			using pointer = const std::shared_ptr<IDBData>*;
			using reference = const std::shared_ptr<IDBData>&;

			Iterator(std::vector<std::shared_ptr<IDBData>>::const_iterator it,
				std::vector<std::shared_ptr<IDBData>>::const_iterator end)
				: m_iterator{ it }
				, m_end{ end }
			{
				SkipNullptr();
			}

			reference operator*() const
			{
				return *m_iterator;
			}

			pointer operator->() const
			{
				return &(*m_iterator);
			}

			Iterator& operator++()
			{
				++m_iterator;
				SkipNullptr();
				return *this;
			}

			Iterator operator++(int)
			{
				Iterator tmp = *this;
				++(*this);
				return tmp;
			}

			bool operator==(const Iterator& other) const
			{
				return m_iterator == other.m_iterator;
			}

			bool operator!=(const Iterator& other) const
			{
				return !(*this == other);
			}

		private:
			void SkipNullptr()
			{
				while (m_iterator != m_end && *m_iterator == nullptr)
				{
					++m_iterator;
				}
			}

			std::vector<std::shared_ptr<IDBData>>::const_iterator m_iterator;
			std::vector<std::shared_ptr<IDBData>>::const_iterator m_end;
		};

		PoolBase(IDBSession* pSession);

		void Initialize(size_t hashSize) override;

		std::shared_ptr<const IDBData> GetBase(DBKey key) const final;

		DBKey Insert(std::shared_ptr<IDBData>&& data) final;
		void Set(std::shared_ptr<IDBData>&& data) final;
		void Delete(DBKey key) final;

		DBKey DirectInsert(std::shared_ptr<IDBData>&& data) final;
		void DirectSet(std::shared_ptr<IDBData>&& data) final;
		void DirectDelete(DBKey key) final;

		size_t GetSize() const final { return m_dataArray.size(); }

		Iterator begin() const
		{
			return Iterator(m_dataArray.begin(), m_dataArray.end());
		}

		Iterator end() const
		{
			return Iterator(m_dataArray.end(), m_dataArray.end());
		}

	protected:
		KeyGenerator m_keyGenerator;
		std::vector<std::shared_ptr<IDBData>> m_dataArray{};
	};
}

#include "HeaderPost.h"