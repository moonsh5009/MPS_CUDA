#pragma once

#include "TypeDef.h"
#include "DeviceBuffer.h"
#include "VKDeviceBuffer.h"

#include <tuple>

namespace mcuda
{
	template<typename value_type>
	class DeviceSingleArray
	{
	public:
		using index_type = mcore::IndexType;

		using device_type = DeviceBuffer<value_type>;
		using host_type = std::vector<value_type>;

		DeviceSingleArray() = default;
		~DeviceSingleArray() = default;
		DeviceSingleArray(const DeviceSingleArray&) = delete;
		DeviceSingleArray(DeviceSingleArray&&) = default;
		DeviceSingleArray& operator=(const DeviceSingleArray&) = delete;
		DeviceSingleArray& operator=(DeviceSingleArray&&) = default;

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

			m_value.Clear();
			return true;
		}

		constexpr bool IsEmpty() const noexcept { return GetBuffer().IsEmpty(); }
		size_t GetSize() const noexcept { return GetBuffer().GetSize(); }

		const device_type& GetBuffer() const noexcept { return m_value; }
		device_type& GetBuffer() noexcept { return m_value; }

	protected:
		index_type InsertImpl(const value_type& host_value)
		{
			const auto currValueSize = static_cast<index_type>(GetSize());
			const auto newValueSize = currValueSize + 1;
			m_value.Resize(newValueSize);
			m_value.CopyFromHost(
				&host_value,
				m_value.GetElementOffset(1),
				m_value.GetElementOffset(currValueSize),
				0);
			return currValueSize;
		}

		void SetImpl(index_type index, const value_type& host_value)
		{
			m_value.CopyFromHost(
				&host_value,
				m_value.GetElementOffset(1),
				m_value.GetElementOffset(index),
				0);
		}

		void RemoveImpl(index_type index)
		{
			const auto lastIndex = static_cast<index_type>(GetBuffer().GetSize()) - 1;
			if (index != lastIndex)
			{
				m_value.CopyFromDevice(
					m_value.GetData() + lastIndex,
					m_value.GetElementOffset(1),
					m_value.GetElementOffset(index),
					0);
			}
			m_value.Resize(lastIndex);
		}

		device_type m_value;
	};

	template<typename value_type>
	class VKDeviceSingleArray
	{
	public:
		using index_type = mcore::IndexType;

		using device_type = VKDeviceBuffer<value_type>;
		using host_type = std::vector<value_type>;

		VKDeviceSingleArray() = default;
		VKDeviceSingleArray(vk::BufferUsageFlags usage)
		{
			Initialize(usage);
		}
		~VKDeviceSingleArray() = default;
		VKDeviceSingleArray(const VKDeviceSingleArray&) = delete;
		VKDeviceSingleArray(VKDeviceSingleArray&&) = default;
		VKDeviceSingleArray& operator=(const VKDeviceSingleArray&) = delete;
		VKDeviceSingleArray& operator=(VKDeviceSingleArray&&) = default;

		void Initialize(vk::BufferUsageFlags usage)
		{
			m_value.Initialize(usage);
		}

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

			m_value.Clear();
			return true;
		}

		constexpr bool IsEmpty() const noexcept { return GetBuffer().IsEmpty(); }
		size_t GetSize() const noexcept { return GetBuffer().GetSize(); }

		const device_type& GetBuffer() const noexcept { return m_value; }
		device_type& GetBuffer() noexcept { return m_value; }

	protected:
		index_type InsertImpl(const value_type& host_value)
		{
			const auto currValueSize = static_cast<index_type>(GetSize());
			const auto newValueSize = currValueSize + 1;
			m_value.Resize(newValueSize);
			m_value.CopyFromHost(
				&host_value,
				m_value.GetElementOffset(1),
				m_value.GetElementOffset(currValueSize),
				0);
			return currValueSize;
		}

		void SetImpl(index_type index, const value_type& host_value)
		{
			m_value.CopyFromHost(
				&host_value,
				m_value.GetElementOffset(1),
				m_value.GetElementOffset(index),
				0);
		}

		void RemoveImpl(index_type index)
		{
			const auto lastIndex = static_cast<index_type>(GetSize()) - 1;
			if (index != lastIndex)
			{
				m_value.CopyFromDevice(
					m_value.GetData() + lastIndex,
					m_value.GetElementOffset(1),
					m_value.GetElementOffset(index),
					0);
			}
			m_value.Resize(lastIndex);
		}

		device_type m_value;
	};
}