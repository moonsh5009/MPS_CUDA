#pragma once

#include "Logger.h"
#include "TypeDef.h"
#include "DeviceBuffer.h"
#include "VKDeviceBuffer.h"

#include <optional>
#include <vector>

namespace mcuda
{
	struct DeviceMultiArrayInfo
	{
		size_t rangeCount;
		mcore::IndexType* rangeOffsets;

		size_t valueCount;
		mcore::IndexType* rangeIndexOfValues;
	};

	template<typename ...value_types>
	class DeviceMultiArray
	{
	public:
		using index_type = mcore::IndexType;
		using device_types = std::tuple<DeviceBuffer<value_types>...>;
		using const_ref_device_types = std::tuple<const DeviceBuffer<value_types>&...>;
		using host_types = std::tuple<std::vector<value_types>...>;
		using const_ref_host_types = std::tuple<const std::vector<value_types>&...>;

		struct Range
		{
			index_type offset;
			index_type size;
		};

		DeviceMultiArray()
		{
			Initialize();
		}
		~DeviceMultiArray() = default;
		DeviceMultiArray(const DeviceMultiArray&) = delete;
		DeviceMultiArray(DeviceMultiArray&&) = default;
		DeviceMultiArray& operator=(const DeviceMultiArray&) = delete;
		DeviceMultiArray& operator=(DeviceMultiArray&&) = default;

		void Initialize()
		{
			std::apply([&](auto&... values)
			{
				(values.Clear(), ...);
			}, m_values);
			m_value_indices.Clear();
			m_offsets.Resize(1, 0);
			m_value_ranges.clear();
		}

		bool Contains(index_type index) const
		{
			return index < GetSize();
		}

		std::optional<Range> GetRange(index_type index) const
		{
			if (index >= GetSize())
				return {};
			return m_value_ranges[index];
		}

		const std::vector<Range>& GetRanges() const
		{
			return m_value_ranges;
		}

		template<typename... HostTypes>
			requires (sizeof...(HostTypes) == sizeof...(value_types))
		&& (... && std::is_same_v<std::remove_cvref_t<HostTypes>, std::vector<value_types>>)
			void Insert(const HostTypes&... host_values)
		{
			auto tuple_values = std::tie(host_values...);
			const auto size = std::get<0>(tuple_values).size();
			if (!std::apply([size](const auto&... arrays) { return (... && (arrays.size() == size)); }, tuple_values))
				return;

			InsertImpl(tuple_values);
		}

		template<typename... HostTypes>
			requires (sizeof...(HostTypes) == sizeof...(value_types))
		&& (... && std::is_same_v<std::remove_cvref_t<HostTypes>, std::vector<value_types>>)
			bool Set(index_type index, const HostTypes&... host_values)
		{
			if (index >= GetSize())
				return false;

			auto tuple_values = std::tie(host_values...);
			const auto size = std::get<0>(tuple_values).size();
			if (!std::apply([size](const auto&... arrays) { return (... && (arrays.size() == size)); }, tuple_values))
				return false;

			return SetImpl(index, tuple_values);
		}

		template<typename... DeviceTypes>
			requires (sizeof...(DeviceTypes) == sizeof...(value_types))
		&& (... && std::is_same_v<std::remove_cvref_t<DeviceTypes>, DeviceBuffer<value_types>>)
			bool Set(index_type index, const DeviceTypes&... device_values)
		{
			if (index >= GetSize())
				return false;

			auto tuple_values = std::tie(device_values...);
			const auto size = std::get<0>(tuple_values).GetSize();
			if (!std::apply([size](const auto&... buffers) { return (... && (buffers.GetSize() == size)); }, tuple_values))
				return false;

			return SetImpl(index, tuple_values);
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

			std::apply([&](auto&... values)
			{
				(values.Clear(), ...);
			}, m_values);
			m_value_indices.Clear();
			m_offsets.Resize(1, 0);
			m_value_ranges.clear();
			return true;
		}

		void DebugPrintInfo() const
		{
			if (IsEmpty())
			{
				mcore::Logger::Debug("DeviceMultiArray: [EMPTY]");
				mcore::Logger::Print();
				return;
			}

			mcore::Logger::Debug("=== DeviceMultiArray Debug Info ===");
			mcore::Logger::Debug("Total ranges: ", GetSize());
			mcore::Logger::Debug("Total value count: ", GetBuffer<0>().GetSize());

			std::tuple<std::vector<value_types>...> hostBuffers;
			[&] <std::size_t... Is>(std::index_sequence<Is...>)
			{
				const auto totalSize = GetBuffer<0>().GetSize();
				if (totalSize == 0)
					return;
				([&]()
				{
					auto& hostVec = std::get<Is>(hostBuffers);
					hostVec.resize(totalSize);
					std::get<Is>(m_values).CopyToHost(
						hostVec.data(),
						std::get<Is>(m_values).GetByteLength(),
						0, 0
					);
				}(), ...);
			}(std::make_index_sequence<sizeof...(value_types)>{});

			std::vector<index_type> hostIndices(m_value_indices.GetSize());
			if (!hostIndices.empty())
			{
				m_value_indices.CopyToHost(
					hostIndices.data(),
					m_value_indices.GetByteLength(),
					0, 0
				);
			}

			std::vector<index_type> hostOffsets(m_offsets.GetSize());
			if (!hostOffsets.empty())
			{
				m_offsets.CopyToHost(
					hostOffsets.data(),
					m_offsets.GetByteLength(),
					0, 0
				);
			}

			mcore::Logger::Debug("--- Ranges ---");
			for (size_t i = 0; i < m_value_ranges.size(); ++i)
			{
				const auto& range = m_value_ranges[i];
				mcore::Logger::Debug("Range[", i, "]: offset=", range.offset, ", size=", range.size);

				if (range.size > 0 && range.offset < GetBuffer<0>().GetSize())
				{
					PrintRangeData(hostBuffers, range.offset, range.size,
						std::make_index_sequence<sizeof...(value_types)>{});
				}
			}

			if (!hostIndices.empty())
			{
				mcore::Logger::Debug("--- Value Indices ---");
				mcore::Logger::DebugNoNewLine("Indices: [");
				const auto maxPrint = std::min<size_t>(hostIndices.size(), 20);
				for (size_t i = 0; i < maxPrint; ++i)
				{
					if (i > 0) mcore::Logger::DebugNoNewLine(", ");
					mcore::Logger::DebugNoNewLine(hostIndices[i]);
				}
				if (hostIndices.size() > maxPrint)
				{
					mcore::Logger::DebugNoNewLine(", ... (", hostIndices.size() - maxPrint, " more)");
				}
				mcore::Logger::Debug("]");
			}

			if (!hostOffsets.empty())
			{
				mcore::Logger::Debug("--- Offsets ---");
				mcore::Logger::DebugNoNewLine("Offsets: [");
				for (size_t i = 0; i < hostOffsets.size(); ++i)
				{
					if (i > 0) mcore::Logger::DebugNoNewLine(", ");
					mcore::Logger::DebugNoNewLine(hostOffsets[i]);
				}
				mcore::Logger::Debug("]");
			}

			mcore::Logger::Debug("=================================");
			mcore::Logger::Print();
		}

		constexpr bool IsEmpty() const noexcept { return GetSize() == 0; }
		constexpr bool HasValue() const noexcept { return GetSize() > 0; }

		size_t GetSize() const noexcept { return m_value_ranges.size(); }

		template<size_t index = 0>
		const std::tuple_element_t<index, device_types>& GetBuffer() const noexcept { return std::get<index>(m_values); }

		template<size_t index = 0>
		std::tuple_element_t<index, device_types>& GetBuffer() noexcept { return std::get<index>(m_values); }

		const DeviceBuffer<index_type>& GetValueIndices() const noexcept { return m_value_indices; }
		const DeviceBuffer<index_type>& GetOffsets() const noexcept { return m_offsets; }

		DeviceMultiArrayInfo GetDeviceInfo() const
		{
			DeviceMultiArrayInfo info;
			info.rangeCount = m_offsets.GetSize() - 1;
			info.rangeOffsets = m_offsets.GetData();
			info.valueCount = m_value_indices.GetSize();
			info.rangeIndexOfValues = m_value_indices.GetData();
			return info;
		}

	protected:
		void InsertImpl(const const_ref_host_types& host_values)
		{
			const auto index = static_cast<index_type>(m_value_ranges.size());
			const auto valueSize = static_cast<index_type>(std::get<0>(host_values).size());
			const auto currValueSize = static_cast<index_type>(GetBuffer().GetSize());
			const auto newValueSize = currValueSize + valueSize;

			if (valueSize > 0)
			{
				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).Resize(newValueSize), ...);
					(std::get<Is>(m_values).CopyFromHost(
						std::get<Is>(host_values).data(),
						std::get<Is>(m_values).GetElementOffset(valueSize),
						std::get<Is>(m_values).GetElementOffset(currValueSize),
						0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});

				// Fill value indices with current index
				m_value_indices.Resize(newValueSize);
				std::vector<index_type> indices(valueSize, index);
				m_value_indices.CopyFromHost(indices.data(),
					m_value_indices.GetElementOffset(valueSize),
					m_value_indices.GetElementOffset(currValueSize),
					0);
			}

			const auto currOffsetSize = static_cast<index_type>(m_offsets.GetSize());
			const auto newOffsetSize = currOffsetSize + 1;
			m_offsets.Resize(newOffsetSize);
			const index_type newOffset = static_cast<index_type>(newValueSize);
			m_offsets.CopyFromHost(&newOffset,
				m_offsets.GetElementOffset(1),
				m_offsets.GetElementOffset(currOffsetSize),
				0);

			m_value_ranges.push_back(Range{ currValueSize, valueSize });
		}

		bool SetImpl(index_type index, const const_ref_host_types& host_values)
		{
			const auto originRange = m_value_ranges[index];
			const auto valueSize = static_cast<index_type>(std::get<0>(host_values).size());

			if (originRange.size == 0 && valueSize == 0)
				return false;

			const auto currValueSize = static_cast<index_type>(GetBuffer().GetSize());
			const auto newValueSize = valueSize > originRange.size
				? currValueSize + (valueSize - originRange.size) : currValueSize - (originRange.size - valueSize);

			// Case 1: Same size - simple replace
			if (valueSize == originRange.size)
			{
				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).CopyFromHost(
					 std::get<Is>(host_values).data(),
					 std::get<Is>(m_values).GetElementOffset(valueSize),
					 std::get<Is>(m_values).GetElementOffset(originRange.offset),
					 0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});
				return true;
			}

			const auto backStartOffset = originRange.offset + originRange.size;
			const auto backSize = currValueSize - backStartOffset;

			// Case 2: No data after this range - resize directly
			if (backSize == 0)
			{
				ResizeArraysForSet(index, valueSize, originRange, newValueSize);
			}
			// Case 3: Has data after this range - need to preserve it
			else
			{
				PreserveAndResize(index, valueSize, originRange, backStartOffset, backSize, newValueSize);
			}

			// Copy new data
			[&] <std::size_t... Is>(std::index_sequence<Is...>)
			{
				(std::get<Is>(m_values).CopyFromHost(
				 std::get<Is>(host_values).data(),
				 std::get<Is>(m_values).GetElementOffset(valueSize),
				 std::get<Is>(m_values).GetElementOffset(originRange.offset),
				 0), ...);
			}(std::make_index_sequence<sizeof...(value_types)>{});

			m_value_ranges[index].size = valueSize;
			return true;
		}

		bool SetImpl(index_type index, const const_ref_device_types& device_values)
		{
			const auto originRange = m_value_ranges[index];
			const auto valueSize = static_cast<index_type>(std::get<0>(device_values).GetSize());

			if (originRange.size == 0 && valueSize == 0)
				return false;

			const auto currValueSize = static_cast<index_type>(GetBuffer().GetSize());
			const auto newValueSize = valueSize > originRange.size
				? currValueSize + (valueSize - originRange.size) : currValueSize - (originRange.size - valueSize);

			// Case 1: Same size - simple replace
			if (valueSize == originRange.size)
			{
				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).CopyFromDevice(
						std::get<Is>(device_values).GetData(),
						std::get<Is>(m_values).GetElementOffset(valueSize),
						std::get<Is>(m_values).GetElementOffset(originRange.offset),
						0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});
				return true;
			}

			const auto backStartOffset = originRange.offset + originRange.size;
			const auto backSize = currValueSize - backStartOffset;

			if (backSize == 0)
			{
				ResizeArraysForSet(index, valueSize, originRange, newValueSize);
			}
			else
			{
				PreserveAndResize(index, valueSize, originRange, backStartOffset, backSize, newValueSize);
			}

			// Copy new data from device
			[&] <std::size_t... Is>(std::index_sequence<Is...>)
			{
				(std::get<Is>(m_values).CopyFromDevice(
					std::get<Is>(device_values).GetData(),
					std::get<Is>(m_values).GetElementOffset(valueSize),
					std::get<Is>(m_values).GetElementOffset(originRange.offset),
					0), ...);
			}(std::make_index_sequence<sizeof...(value_types)>{});

			m_value_ranges[index].size = valueSize;
			return true;
		}

		void RemoveImpl(index_type index)
		{
			const auto& removedRange = m_value_ranges[index];
			const auto& lastRange = m_value_ranges.back();

			const auto currValueSize = static_cast<index_type>(GetBuffer().GetSize());
			const auto newValueSize = currValueSize - removedRange.size;

			// Case 1: Removing last element
			if (index == static_cast<index_type>(m_value_ranges.size()) - 1)
			{
				std::apply([&](auto&... values)
				{
					(values.Resize(newValueSize), ...);
				}, m_values);
				m_value_indices.Resize(newValueSize);
				m_offsets.Resize(static_cast<index_type>(m_offsets.GetSize()) - 1);
				m_value_ranges.pop_back();
				return;
			}

			// Case 2: Same size - simple swap
			if (removedRange.size == lastRange.size && lastRange.size > 0)
			{
				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).CopyFromDevice(
						std::get<Is>(m_values).GetData() + lastRange.offset,
						std::get<Is>(m_values).GetElementOffset(lastRange.size),
						std::get<Is>(m_values).GetElementOffset(removedRange.offset),
						0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});

				std::vector<index_type> lastIndices(lastRange.size, static_cast<index_type>(m_value_ranges.size() - 1));
				m_value_indices.CopyFromHost(lastIndices.data(),
					m_value_indices.GetElementOffset(lastRange.size),
					m_value_indices.GetElementOffset(removedRange.offset),
					0);
			}

			std::apply([&](auto&... values)
			{
				(values.Resize(newValueSize), ...);
			}, m_values);
			m_value_indices.Resize(newValueSize);
			m_offsets.Resize(static_cast<index_type>(m_offsets.GetSize()) - 1);

			// Update offsets
			UpdateOffsetsAfterRemove(index, removedRange.size);

			m_value_ranges[index].offset = removedRange.offset;
			m_value_ranges[index].size = lastRange.size;
			m_value_ranges.pop_back();
		}

		void ResizeArraysForSet(index_type index, index_type newSize, const Range& originRange, index_type totalNewSize)
		{
			if (newSize > originRange.size)
			{
				const auto gapSize = newSize - originRange.size;
				std::apply([&](auto&... values)
				{
					(values.Resize(totalNewSize), ...);
				}, m_values);
				m_value_indices.Resize(totalNewSize);

				// Fill gap with current index
				std::vector<index_type> newIndices(gapSize, index);
				m_value_indices.CopyFromHost(newIndices.data(),
					m_value_indices.GetElementOffset(gapSize),
					m_value_indices.GetElementOffset(originRange.offset + originRange.size),
					0);

				// Update ranges after this index
				for (size_t i = index + 1; i < m_value_ranges.size(); ++i)
				{
					m_value_ranges[i].offset += gapSize;
				}
			}
			else
			{
				const auto gapSize = originRange.size - newSize;
				std::apply([&](auto&... values)
				{
					(values.Resize(totalNewSize), ...);
				}, m_values);
				m_value_indices.Resize(totalNewSize);

				for (size_t i = index + 1; i < m_value_ranges.size(); ++i)
				{
					m_value_ranges[i].offset -= gapSize;
				}
			}

			// Update last offset
			const auto lastOffset = static_cast<index_type>(GetBuffer().GetSize());
			const auto lastOffsetIndex = static_cast<index_type>(m_offsets.GetSize()) - 1;
			m_offsets.CopyFromHost(&lastOffset,
				m_offsets.GetElementOffset(1),
				m_offsets.GetElementOffset(lastOffsetIndex),
				0);
		}

		void PreserveAndResize(index_type index, index_type newSize, const Range& originRange,
			index_type backStartOffset, index_type backSize, index_type totalNewSize)
		{
			// Backup data after the range
			device_types backValues{ DeviceBuffer<value_types>{}... };
			DeviceBuffer<index_type> backValueIndices;

			[&] <std::size_t... Is>(std::index_sequence<Is...>)
			{
				(std::get<Is>(backValues).Resize(backSize), ...);
				(std::get<Is>(backValues).CopyFromDevice(
					std::get<Is>(m_values).GetData() + backStartOffset,
					std::get<Is>(backValues).GetElementOffset(backSize),
					0), ...);
			}(std::make_index_sequence<sizeof...(value_types)>{});

			backValueIndices.Resize(backSize);
			backValueIndices.CopyFromDevice(
				m_value_indices.GetData() + backStartOffset,
				backValueIndices.GetElementOffset(backSize),
				0);

			// Resize and restore
			if (newSize > originRange.size)
			{
				const auto gapSize = newSize - originRange.size;
				const auto pasteOffset = backStartOffset + gapSize;

				std::apply([&](auto&... values)
				{
					(values.Resize(totalNewSize), ...);
				}, m_values);
				m_value_indices.Resize(totalNewSize);

				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).CopyFromDevice(
						std::get<Is>(backValues).GetData(),
						std::get<Is>(m_values).GetElementOffset(backSize),
						std::get<Is>(m_values).GetElementOffset(pasteOffset),
						0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});

				m_value_indices.CopyFromDevice(
					backValueIndices.GetData(),
					m_value_indices.GetElementOffset(backSize),
					m_value_indices.GetElementOffset(pasteOffset),
					0);

				// Fill gap
				std::vector<index_type> newIndices(gapSize, index);
				m_value_indices.CopyFromHost(newIndices.data(),
					m_value_indices.GetElementOffset(gapSize),
					m_value_indices.GetElementOffset(originRange.offset + originRange.size),
					0);

				// Update offsets
				UpdateOffsetsAfterResize(index, gapSize, true);

				for (size_t i = index + 1; i < m_value_ranges.size(); ++i)
				{
					m_value_ranges[i].offset += gapSize;
				}
			}
			else
			{
				const auto gapSize = originRange.size - newSize;
				const auto pasteOffset = backStartOffset - gapSize;

				std::apply([&](auto&... values)
				{
					(values.Resize(totalNewSize), ...);
				}, m_values);
				m_value_indices.Resize(totalNewSize);

				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).CopyFromDevice(
						std::get<Is>(backValues).GetData(),
						std::get<Is>(m_values).GetElementOffset(backSize),
						std::get<Is>(m_values).GetElementOffset(pasteOffset),
						0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});

				m_value_indices.CopyFromDevice(
					backValueIndices.GetData(),
					m_value_indices.GetElementOffset(backSize),
					m_value_indices.GetElementOffset(pasteOffset),
					0);

				UpdateOffsetsAfterResize(index, gapSize, false);

				for (size_t i = index + 1; i < m_value_ranges.size(); ++i)
				{
					m_value_ranges[i].offset -= gapSize;
				}
			}
		}

		void UpdateOffsetsAfterResize(index_type startIndex, index_type delta, bool isIncrease)
		{
			const auto offsetSize = static_cast<index_type>(m_offsets.GetSize());
			std::vector<index_type> offsets(offsetSize);
			m_offsets.CopyToHost(offsets.data(), m_offsets.GetByteLength(), 0, 0);

			for (size_t i = startIndex + 1; i < offsetSize; ++i)
			{
				if (isIncrease)
					offsets[i] += delta;
				else
					offsets[i] -= delta;
			}

			m_offsets.CopyFromHost(offsets.data() + startIndex + 1,
				m_offsets.GetElementOffset(offsetSize - startIndex - 1),
				m_offsets.GetElementOffset(startIndex + 1),
				0);
		}

		void UpdateOffsetsAfterRemove(index_type removedIndex, index_type removedSize)
		{
			const auto offsetSize = static_cast<index_type>(m_offsets.GetSize());
			std::vector<index_type> offsets(offsetSize + 1);
			m_offsets.CopyToHost(offsets.data(), m_offsets.GetByteLength(), 0, 0);

			// Remove one offset and adjust remaining
			for (size_t i = removedIndex + 1; i < offsetSize; ++i)
			{
				offsets[i] = offsets[i + 1] - removedSize;
			}

			m_offsets.CopyFromHost(offsets.data() + removedIndex + 1,
				m_offsets.GetElementOffset(offsetSize - removedIndex - 1),
				m_offsets.GetElementOffset(removedIndex + 1),
				0);
		}

		template<std::size_t... Is>
		void PrintRangeData(const std::tuple<std::vector<value_types>...>& hostBuffers,
			index_type offset, index_type size,
			std::index_sequence<Is...>) const
		{
			const auto printCount = std::min<size_t>(size, 10);
			mcore::Logger::Debug("  Data (showing ", printCount, " of ", size, " elements):");

			for (size_t i = 0; i < printCount; ++i)
			{
				const auto idx = offset + i;
				mcore::Logger::DebugNoNewLine("    [", i, "]: ");

				size_t tupleIdx = 0;
				([&]()
				{
					const auto& vec = std::get<Is>(hostBuffers);
					if (idx < vec.size())
					{
						if (tupleIdx > 0) mcore::Logger::DebugNoNewLine(", ");
						mcore::Logger::DebugNoNewLine(vec[idx]);
						++tupleIdx;
					}
				}(), ...);

				mcore::Logger::Debug("");
			}

			if (size > printCount)
			{
				mcore::Logger::Debug("    ... (", size - printCount, " more elements)");
			}
		}

		std::vector<Range> m_value_ranges;
		device_types m_values;
		DeviceBuffer<index_type> m_value_indices;
		DeviceBuffer<index_type> m_offsets;
	};

	template<typename ...value_types>
	class VKDeviceMultiArray
	{
	public:
		using index_type = mcore::IndexType;
		using device_types = std::tuple<VKDeviceBuffer<value_types>...>;
		using const_ref_device_types = std::tuple<const VKDeviceBuffer<value_types>&...>;
		using host_types = std::tuple<std::vector<value_types>...>;
		using const_ref_host_types = std::tuple<const std::vector<value_types>&...>;

		struct Range
		{
			index_type offset;
			index_type size;
		};

		VKDeviceMultiArray() = default;
		VKDeviceMultiArray(vk::BufferUsageFlags usage)
		{
			Initialize(usage);
		}
		~VKDeviceMultiArray() = default;
		VKDeviceMultiArray(const VKDeviceMultiArray&) = delete;
		VKDeviceMultiArray(VKDeviceMultiArray&&) = default;
		VKDeviceMultiArray& operator=(const VKDeviceMultiArray&) = delete;
		VKDeviceMultiArray& operator=(VKDeviceMultiArray&&) = default;

		void Initialize(vk::BufferUsageFlags usage)
		{
			std::apply([&](auto&... values)
			{
				(values.Initialize(usage), ...);
				(values.Clear(), ...);
			}, m_values);

			m_value_indices.Initialize(vk::BufferUsageFlagBits::eStorageBuffer);
			m_offsets.Initialize(vk::BufferUsageFlagBits::eStorageBuffer);
			m_value_indices.Clear();
			m_offsets.Resize(1, 0);
			m_value_ranges.clear();
		}

		bool Contains(index_type index) const
		{
			return index < GetSize();
		}

		std::optional<Range> GetRange(index_type index) const
		{
			if (index >= GetSize())
				return {};
			return m_value_ranges[index];
		}

		const std::vector<Range>& GetRanges() const
		{
			return m_value_ranges;
		}

		template<typename... HostTypes>
			requires (sizeof...(HostTypes) == sizeof...(value_types))
		&& (... && std::is_same_v<std::remove_cvref_t<HostTypes>, std::vector<value_types>>)
			void Insert(const HostTypes&... host_values)
		{
			auto tuple_values = std::tie(host_values...);
			const auto size = std::get<0>(tuple_values).size();
			if (!std::apply([size](const auto&... arrays) { return (... && (arrays.size() == size)); }, tuple_values))
				return;

			InsertImpl(tuple_values);
		}

		template<typename... HostTypes>
			requires (sizeof...(HostTypes) == sizeof...(value_types))
		&& (... && std::is_same_v<std::remove_cvref_t<HostTypes>, std::vector<value_types>>)
			bool Set(index_type index, const HostTypes&... host_values)
		{
			if (index >= GetSize())
				return false;

			auto tuple_values = std::tie(host_values...);
			const auto size = std::get<0>(tuple_values).size();
			if (!std::apply([size](const auto&... arrays) { return (... && (arrays.size() == size)); }, tuple_values))
				return false;

			return SetImpl(index, tuple_values);
		}

		template<typename... DeviceTypes>
			requires (sizeof...(DeviceTypes) == sizeof...(value_types))
		&& (... && std::is_same_v<std::remove_cvref_t<DeviceTypes>, VKDeviceBuffer<value_types>>)
			bool Set(index_type index, const DeviceTypes&... device_values)
		{
			if (index >= GetSize())
				return false;

			auto tuple_values = std::tie(device_values...);
			const auto size = std::get<0>(tuple_values).GetSize();
			if (!std::apply([size](const auto&... buffers) { return (... && (buffers.GetSize() == size)); }, tuple_values))
				return false;

			return SetImpl(index, tuple_values);
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

			std::apply([&](auto&... values)
			{
				(values.Clear(), ...);
			}, m_values);
			m_value_indices.Clear();
			m_offsets.Resize(1, 0);
			m_value_ranges.clear();
			return true;
		}

		void DebugPrintInfo() const
		{
			if (IsEmpty())
			{
				mcore::Logger::Debug("DeviceMultiArray: [EMPTY]");
				mcore::Logger::Print();
				return;
			}

			mcore::Logger::Debug("=== DeviceMultiArray Debug Info ===");
			mcore::Logger::Debug("Total ranges: ", GetSize());
			mcore::Logger::Debug("Total value count: ", GetBuffer<0>().GetSize());

			std::tuple<std::vector<value_types>...> hostBuffers;
			[&] <std::size_t... Is>(std::index_sequence<Is...>)
			{
				const auto totalSize = GetBuffer<0>().GetSize();
				if (totalSize == 0)
					return;
				([&]()
				{
					auto& hostVec = std::get<Is>(hostBuffers);
					hostVec.resize(totalSize);
					std::get<Is>(m_values).CopyToHost(
						hostVec.data(),
						std::get<Is>(m_values).GetByteLength(),
						0, 0
					);
				}(), ...);
			}(std::make_index_sequence<sizeof...(value_types)>{});

			std::vector<index_type> hostIndices(m_value_indices.GetSize());
			if (!hostIndices.empty())
			{
				m_value_indices.CopyToHost(
					hostIndices.data(),
					m_value_indices.GetByteLength(),
					0, 0
				);
			}

			std::vector<index_type> hostOffsets(m_offsets.GetSize());
			if (!hostOffsets.empty())
			{
				m_offsets.CopyToHost(
					hostOffsets.data(),
					m_offsets.GetByteLength(),
					0, 0
				);
			}

			mcore::Logger::Debug("--- Ranges ---");
			for (size_t i = 0; i < m_value_ranges.size(); ++i)
			{
				const auto& range = m_value_ranges[i];
				mcore::Logger::Debug("Range[", i, "]: offset=", range.offset, ", size=", range.size);

				if (range.size > 0 && range.offset < GetBuffer<0>().GetSize())
				{
					PrintRangeData(hostBuffers, range.offset, range.size,
						std::make_index_sequence<sizeof...(value_types)>{});
				}
			}

			if (!hostIndices.empty())
			{
				mcore::Logger::Debug("--- Value Indices ---");
				mcore::Logger::DebugNoNewLine("Indices: [");
				const auto maxPrint = std::min<size_t>(hostIndices.size(), 20);
				for (size_t i = 0; i < maxPrint; ++i)
				{
					if (i > 0) mcore::Logger::DebugNoNewLine(", ");
					mcore::Logger::DebugNoNewLine(hostIndices[i]);
				}
				if (hostIndices.size() > maxPrint)
				{
					mcore::Logger::DebugNoNewLine(", ... (", hostIndices.size() - maxPrint, " more)");
				}
				mcore::Logger::Debug("]");
			}

			if (!hostOffsets.empty())
			{
				mcore::Logger::Debug("--- Offsets ---");
				mcore::Logger::DebugNoNewLine("Offsets: [");
				for (size_t i = 0; i < hostOffsets.size(); ++i)
				{
					if (i > 0) mcore::Logger::DebugNoNewLine(", ");
					mcore::Logger::DebugNoNewLine(hostOffsets[i]);
				}
				mcore::Logger::Debug("]");
			}

			mcore::Logger::Debug("=================================");
			mcore::Logger::Print();
		}

		constexpr bool IsEmpty() const noexcept { return GetSize() == 0; }
		constexpr bool HasValue() const noexcept { return GetSize() > 0; }

		size_t GetSize() const noexcept { return m_value_ranges.size(); }

		template<size_t index = 0>
		const std::tuple_element_t<index, device_types>& GetBuffer() const noexcept { return std::get<index>(m_values); }

		template<size_t index = 0>
		std::tuple_element_t<index, device_types>& GetBuffer() noexcept { return std::get<index>(m_values); }

		const VKDeviceBuffer<index_type>& GetValueIndices() const noexcept { return m_value_indices; }
		const VKDeviceBuffer<index_type>& GetOffsets() const noexcept { return m_offsets; }

		DeviceMultiArrayInfo GetDeviceInfo() const
		{
			DeviceMultiArrayInfo info;
			info.rangeCount = m_offsets.GetSize() - 1;
			info.rangeOffsets = m_offsets.GetData();
			info.valueCount = m_value_indices.GetSize();
			info.rangeIndexOfValues = m_value_indices.GetData();
			return info;
		}

	protected:
		void InsertImpl(const const_ref_host_types& host_values)
		{
			const auto index = static_cast<index_type>(m_value_ranges.size());
			const auto valueSize = static_cast<index_type>(std::get<0>(host_values).size());
			const auto currValueSize = static_cast<index_type>(GetBuffer().GetSize());
			const auto newValueSize = currValueSize + valueSize;

			if (valueSize > 0)
			{
				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).Resize(newValueSize), ...);
					(std::get<Is>(m_values).CopyFromHost(
						std::get<Is>(host_values).data(),
						std::get<Is>(m_values).GetElementOffset(valueSize),
						std::get<Is>(m_values).GetElementOffset(currValueSize),
						0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});

				// Fill value indices with current index
				m_value_indices.Resize(newValueSize);
				std::vector<index_type> indices(valueSize, index);
				m_value_indices.CopyFromHost(indices.data(),
					m_value_indices.GetElementOffset(valueSize),
					m_value_indices.GetElementOffset(currValueSize),
					0);
			}

			const auto currOffsetSize = static_cast<index_type>(m_offsets.GetSize());
			const auto newOffsetSize = currOffsetSize + 1;
			m_offsets.Resize(newOffsetSize);
			const index_type newOffset = static_cast<index_type>(newValueSize);
			m_offsets.CopyFromHost(&newOffset,
				m_offsets.GetElementOffset(1),
				m_offsets.GetElementOffset(currOffsetSize),
				0);

			m_value_ranges.push_back(Range{ currValueSize, valueSize });
		}

		bool SetImpl(index_type index, const const_ref_host_types& host_values)
		{
			const auto originRange = m_value_ranges[index];
			const auto valueSize = static_cast<index_type>(std::get<0>(host_values).size());

			if (originRange.size == 0 && valueSize == 0)
				return false;

			const auto currValueSize = static_cast<index_type>(GetBuffer().GetSize());
			const auto newValueSize = valueSize > originRange.size
				? currValueSize + (valueSize - originRange.size) : currValueSize - (originRange.size - valueSize);

			// Case 1: Same size - simple replace
			if (valueSize == originRange.size)
			{
				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).CopyFromHost(
					 std::get<Is>(host_values).data(),
					 std::get<Is>(m_values).GetElementOffset(valueSize),
					 std::get<Is>(m_values).GetElementOffset(originRange.offset),
					 0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});
				return true;
			}

			const auto backStartOffset = originRange.offset + originRange.size;
			const auto backSize = currValueSize - backStartOffset;

			// Case 2: No data after this range - resize directly
			if (backSize == 0)
			{
				ResizeArraysForSet(index, valueSize, originRange, newValueSize);
			}
			// Case 3: Has data after this range - need to preserve it
			else
			{
				PreserveAndResize(index, valueSize, originRange, backStartOffset, backSize, newValueSize);
			}

			// Copy new data
			[&] <std::size_t... Is>(std::index_sequence<Is...>)
			{
				(std::get<Is>(m_values).CopyFromHost(
				 std::get<Is>(host_values).data(),
				 std::get<Is>(m_values).GetElementOffset(valueSize),
				 std::get<Is>(m_values).GetElementOffset(originRange.offset),
				 0), ...);
			}(std::make_index_sequence<sizeof...(value_types)>{});

			m_value_ranges[index].size = valueSize;
			return true;
		}

		bool SetImpl(index_type index, const const_ref_device_types& device_values)
		{
			const auto originRange = m_value_ranges[index];
			const auto valueSize = static_cast<index_type>(std::get<0>(device_values).GetSize());

			if (originRange.size == 0 && valueSize == 0)
				return false;

			const auto currValueSize = static_cast<index_type>(GetBuffer().GetSize());
			const auto newValueSize = valueSize > originRange.size
				? currValueSize + (valueSize - originRange.size) : currValueSize - (originRange.size - valueSize);

			// Case 1: Same size - simple replace
			if (valueSize == originRange.size)
			{
				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).CopyFromDevice(
						std::get<Is>(device_values).GetData(),
						std::get<Is>(m_values).GetElementOffset(valueSize),
						std::get<Is>(m_values).GetElementOffset(originRange.offset),
						0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});
				return true;
			}

			const auto backStartOffset = originRange.offset + originRange.size;
			const auto backSize = currValueSize - backStartOffset;

			if (backSize == 0)
			{
				ResizeArraysForSet(index, valueSize, originRange, newValueSize);
			}
			else
			{
				PreserveAndResize(index, valueSize, originRange, backStartOffset, backSize, newValueSize);
			}

			// Copy new data from device
			[&] <std::size_t... Is>(std::index_sequence<Is...>)
			{
				(std::get<Is>(m_values).CopyFromDevice(
					std::get<Is>(device_values).GetData(),
					std::get<Is>(m_values).GetElementOffset(valueSize),
					std::get<Is>(m_values).GetElementOffset(originRange.offset),
					0), ...);
			}(std::make_index_sequence<sizeof...(value_types)>{});

			m_value_ranges[index].size = valueSize;
			return true;
		}

		void RemoveImpl(index_type index)
		{
			const auto& removedRange = m_value_ranges[index];
			const auto& lastRange = m_value_ranges.back();

			const auto currValueSize = static_cast<index_type>(GetBuffer().GetSize());
			const auto newValueSize = currValueSize - removedRange.size;

			// Case 1: Removing last element
			if (index == static_cast<index_type>(m_value_ranges.size()) - 1)
			{
				std::apply([&](auto&... values)
				{
					(values.Resize(newValueSize), ...);
				}, m_values);
				m_value_indices.Resize(newValueSize);
				m_offsets.Resize(static_cast<index_type>(m_offsets.GetSize()) - 1);
				m_value_ranges.pop_back();
				return;
			}

			// Case 2: Same size - simple swap
			if (removedRange.size == lastRange.size && lastRange.size > 0)
			{
				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).CopyFromDevice(
						std::get<Is>(m_values).GetData() + lastRange.offset,
						std::get<Is>(m_values).GetElementOffset(lastRange.size),
						std::get<Is>(m_values).GetElementOffset(removedRange.offset),
						0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});

				std::vector<index_type> lastIndices(lastRange.size, static_cast<index_type>(m_value_ranges.size() - 1));
				m_value_indices.CopyFromHost(lastIndices.data(),
					m_value_indices.GetElementOffset(lastRange.size),
					m_value_indices.GetElementOffset(removedRange.offset),
					0);
			}

			std::apply([&](auto&... values)
			{
				(values.Resize(newValueSize), ...);
			}, m_values);
			m_value_indices.Resize(newValueSize);
			m_offsets.Resize(static_cast<index_type>(m_offsets.GetSize()) - 1);

			// Update offsets
			UpdateOffsetsAfterRemove(index, removedRange.size);

			m_value_ranges[index].offset = removedRange.offset;
			m_value_ranges[index].size = lastRange.size;
			m_value_ranges.pop_back();
		}

		void ResizeArraysForSet(index_type index, index_type newSize, const Range& originRange, index_type totalNewSize)
		{
			if (newSize > originRange.size)
			{
				const auto gapSize = newSize - originRange.size;
				std::apply([&](auto&... values)
				{
					(values.Resize(totalNewSize), ...);
				}, m_values);
				m_value_indices.Resize(totalNewSize);

				// Fill gap with current index
				std::vector<index_type> newIndices(gapSize, index);
				m_value_indices.CopyFromHost(newIndices.data(),
					m_value_indices.GetElementOffset(gapSize),
					m_value_indices.GetElementOffset(originRange.offset + originRange.size),
					0);

				// Update ranges after this index
				for (size_t i = index + 1; i < m_value_ranges.size(); ++i)
				{
					m_value_ranges[i].offset += gapSize;
				}
			}
			else
			{
				const auto gapSize = originRange.size - newSize;
				std::apply([&](auto&... values)
				{
					(values.Resize(totalNewSize), ...);
				}, m_values);
				m_value_indices.Resize(totalNewSize);

				for (size_t i = index + 1; i < m_value_ranges.size(); ++i)
				{
					m_value_ranges[i].offset -= gapSize;
				}
			}

			// Update last offset
			const auto lastOffset = static_cast<index_type>(GetBuffer().GetSize());
			const auto lastOffsetIndex = static_cast<index_type>(m_offsets.GetSize()) - 1;
			m_offsets.CopyFromHost(&lastOffset,
				m_offsets.GetElementOffset(1),
				m_offsets.GetElementOffset(lastOffsetIndex),
				0);
		}

		void PreserveAndResize(index_type index, index_type newSize, const Range& originRange,
			index_type backStartOffset, index_type backSize, index_type totalNewSize)
		{
			// Backup data after the range
			device_types backValues{ VKDeviceBuffer<value_types>{}... };
			VKDeviceBuffer<index_type> backValueIndices;

			[&] <std::size_t... Is>(std::index_sequence<Is...>)
			{
				(std::get<Is>(backValues).Resize(backSize), ...);
				(std::get<Is>(backValues).CopyFromDevice(
					std::get<Is>(m_values).GetData() + backStartOffset,
					std::get<Is>(backValues).GetElementOffset(backSize),
					0), ...);
			}(std::make_index_sequence<sizeof...(value_types)>{});

			backValueIndices.Resize(backSize);
			backValueIndices.CopyFromDevice(
				m_value_indices.GetData() + backStartOffset,
				backValueIndices.GetElementOffset(backSize),
				0);

			// Resize and restore
			if (newSize > originRange.size)
			{
				const auto gapSize = newSize - originRange.size;
				const auto pasteOffset = backStartOffset + gapSize;

				std::apply([&](auto&... values)
				{
					(values.Resize(totalNewSize), ...);
				}, m_values);
				m_value_indices.Resize(totalNewSize);

				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).CopyFromDevice(
						std::get<Is>(backValues).GetData(),
						std::get<Is>(m_values).GetElementOffset(backSize),
						std::get<Is>(m_values).GetElementOffset(pasteOffset),
						0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});

				m_value_indices.CopyFromDevice(
					backValueIndices.GetData(),
					m_value_indices.GetElementOffset(backSize),
					m_value_indices.GetElementOffset(pasteOffset),
					0);

				// Fill gap
				std::vector<index_type> newIndices(gapSize, index);
				m_value_indices.CopyFromHost(newIndices.data(),
					m_value_indices.GetElementOffset(gapSize),
					m_value_indices.GetElementOffset(originRange.offset + originRange.size),
					0);

				// Update offsets
				UpdateOffsetsAfterResize(index, gapSize, true);

				for (size_t i = index + 1; i < m_value_ranges.size(); ++i)
				{
					m_value_ranges[i].offset += gapSize;
				}
			}
			else
			{
				const auto gapSize = originRange.size - newSize;
				const auto pasteOffset = backStartOffset - gapSize;

				std::apply([&](auto&... values)
				{
					(values.Resize(totalNewSize), ...);
				}, m_values);
				m_value_indices.Resize(totalNewSize);

				[&] <std::size_t... Is>(std::index_sequence<Is...>)
				{
					(std::get<Is>(m_values).CopyFromDevice(
						std::get<Is>(backValues).GetData(),
						std::get<Is>(m_values).GetElementOffset(backSize),
						std::get<Is>(m_values).GetElementOffset(pasteOffset),
						0), ...);
				}(std::make_index_sequence<sizeof...(value_types)>{});

				m_value_indices.CopyFromDevice(
					backValueIndices.GetData(),
					m_value_indices.GetElementOffset(backSize),
					m_value_indices.GetElementOffset(pasteOffset),
					0);

				UpdateOffsetsAfterResize(index, gapSize, false);

				for (size_t i = index + 1; i < m_value_ranges.size(); ++i)
				{
					m_value_ranges[i].offset -= gapSize;
				}
			}
		}

		void UpdateOffsetsAfterResize(index_type startIndex, index_type delta, bool isIncrease)
		{
			const auto offsetSize = static_cast<index_type>(m_offsets.GetSize());
			std::vector<index_type> offsets(offsetSize);
			m_offsets.CopyToHost(offsets.data(), m_offsets.GetByteLength(), 0, 0);

			for (size_t i = startIndex + 1; i < offsetSize; ++i)
			{
				if (isIncrease)
					offsets[i] += delta;
				else
					offsets[i] -= delta;
			}

			m_offsets.CopyFromHost(offsets.data() + startIndex + 1,
				m_offsets.GetElementOffset(offsetSize - startIndex - 1),
				m_offsets.GetElementOffset(startIndex + 1),
				0);
		}

		void UpdateOffsetsAfterRemove(index_type removedIndex, index_type removedSize)
		{
			const auto offsetSize = static_cast<index_type>(m_offsets.GetSize());
			std::vector<index_type> offsets(offsetSize + 1);
		 m_offsets.CopyToHost(offsets.data(), m_offsets.GetByteLength(), 0, 0);

			// Remove one offset and adjust remaining
			for (size_t i = removedIndex + 1; i < offsetSize; ++i)
			{
				offsets[i] = offsets[i + 1] - removedSize;
			}

			m_offsets.CopyFromHost(offsets.data() + removedIndex + 1,
				m_offsets.GetElementOffset(offsetSize - removedIndex - 1),
				m_offsets.GetElementOffset(removedIndex + 1),
				0);
		}

		template<std::size_t... Is>
		void PrintRangeData(const std::tuple<std::vector<value_types>...>& hostBuffers,
			index_type offset, index_type size,
			std::index_sequence<Is...>) const
		{
			const auto printCount = std::min<size_t>(size, 10);
			mcore::Logger::Debug("  Data (showing ", printCount, " of ", size, " elements):");

			for (size_t i = 0; i < printCount; ++i)
			{
				const auto idx = offset + i;
				mcore::Logger::DebugNoNewLine("    [", i, "]: ");

				size_t tupleIdx = 0;
				([&]()
				{
					const auto& vec = std::get<Is>(hostBuffers);
					if (idx < vec.size())
					{
						if (tupleIdx > 0) mcore::Logger::DebugNoNewLine(", ");
						mcore::Logger::DebugNoNewLine(vec[idx]);
						++tupleIdx;
					}
				}(), ...);

				mcore::Logger::Debug("");
			}

			if (size > printCount)
			{
				mcore::Logger::Debug("    ... (", size - printCount, " more elements)");
			}
		}

		std::vector<Range> m_value_ranges;
		device_types m_values;
		VKDeviceBuffer<index_type> m_value_indices;
		VKDeviceBuffer<index_type> m_offsets;
	};
}