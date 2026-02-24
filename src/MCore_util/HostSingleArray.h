#pragma once

#include "TypeDef.h"
#include "DeviceBuffer.h"
#include "VKDeviceBuffer.h"

#include <tuple>

namespace mcuda
{
	template<typename value_type>
	class HostSingleArray
	{
	public:
		using index_type = mcore::IndexType;

		using host_array = std::vector<value_type>;

		HostSingleArray() = default;
		~HostSingleArray() = default;
		HostSingleArray(const HostSingleArray&) = delete;
		HostSingleArray(HostSingleArray&&) = default;
		HostSingleArray& operator=(const HostSingleArray&) = delete;
		HostSingleArray& operator=(HostSingleArray&&) = default;

		index_type Insert(const value_type& host_value)
		{
			return InsertImpl(host_value);
		}

		bool Set(index_type index, const value_type& host_value)
		{
			if (index >= GetSize())
				return false;
			SetImpl(index, host_value);
			return true;
		}

		bool Remove(index_type index)
		{
			if (index >= GetSize())
				return false;

			RemoveImpl(index);
			return true;
		}

		bool Clear()
		{
			if (IsEmpty())
				return false;

			m_value.clear();
			return true;
		}

		value_type& operator[](index_type index) noexcept { return m_value[index]; }
		const value_type& operator[](index_type index) const noexcept { return m_value[index]; }

		constexpr bool IsEmpty() const noexcept { return GetBuffer().empty(); }
		size_t GetSize() const noexcept { return GetBuffer().size(); }

		const host_array& GetBuffer() const noexcept { return m_value; }
		host_array& GetBuffer() noexcept { return m_value; }

	protected:
		index_type InsertImpl(const value_type& host_value)
		{
			const auto currValueSize = static_cast<index_type>(GetSize());
			m_value.push_back(host_value);
			return currValueSize;
		}

		void SetImpl(index_type index, const value_type& host_value)
		{
			m_value[index] = host_value;
		}

		void RemoveImpl(index_type index)
		{
			const auto lastIndex = static_cast<index_type>(GetBuffer().size()) - 1;
			if (index != lastIndex)
			{
				m_value[index] = m_value[lastIndex];
			}
			m_value.pop_back();
		}

		host_array m_value;
	};
}