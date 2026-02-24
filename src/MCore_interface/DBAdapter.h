#pragma once

#include <type_traits>
#include <array>
#include <vector>
#include <string>

#include <eigen/Eigen/Dense>
#include <glm/glm.hpp>

#include "../MCore_util/BufferDef.h"

template<typename T>
struct DBAdapter : std::false_type
{};

namespace mcore
{
	template <typename T>
	concept DBSerializable = !std::is_base_of_v<std::false_type, DBAdapter<T>>;
}

template<typename T>
	requires std::is_arithmetic_v<T>
		|| std::is_same_v<T, mvk::vbo::PointAttribute>
		|| std::is_same_v<T, mvk::vbo::LineAttribute>
		|| std::is_same_v<T, mvk::vbo::TriangleAttribute>
struct DBAdapter<T>
{
	static size_t GetStreamSize(const T& data)
	{
		return sizeof(T);
	}
	static std::byte* ToStream(std::byte* pStream, const T& data)
	{
		const auto streamSize = GetStreamSize(data);
		std::memcpy(pStream, &data, streamSize);
		return pStream + streamSize;
	}
	static const std::byte* FromStream(const std::byte* pStream, T& data)
	{
		const auto streamSize = GetStreamSize(data);
		std::memcpy(&data, pStream, streamSize);
		return pStream + streamSize;
	}
};

template<typename T>
struct DBAdapter<std::basic_string<T>>
{
	static size_t GetStreamSize(const std::basic_string<T>& data)
	{
		return sizeof(size_t) + data.size() * sizeof(T);
	}
	static std::byte* ToStream(std::byte* pStream, const std::basic_string<T>& data)
	{
		const auto size = data.size();
		std::memcpy(pStream, &size, sizeof(size_t));
		pStream += sizeof(size_t);
		
		if (size > 0)
		{
			std::memcpy(pStream, data.data(), size * sizeof(T));
			pStream += size * sizeof(T);
		}
		return pStream;
	}
	static const std::byte* FromStream(const std::byte* pStream, std::basic_string<T>& data)
	{
		size_t size = 0;
		std::memcpy(&size, pStream, sizeof(size_t));
		pStream += sizeof(size_t);
		
		data.resize(size);
		if (size > 0)
		{
			std::memcpy(data.data(), pStream, size * sizeof(T));
			pStream += size * sizeof(T);
		}
		return pStream;
	}
};

template<uint32_t N, typename T, glm::qualifier Q>
struct DBAdapter<glm::vec<N, T, Q>>
{
	static size_t GetStreamSize(const glm::vec<N, T, Q>& data)
	{
		return sizeof(glm::vec<N, T, Q>);
	}
	static std::byte* ToStream(std::byte* pStream, const glm::vec<N, T, Q>& data)
	{
		const auto streamSize = GetStreamSize(data);
		std::memcpy(pStream, &data, streamSize);
		return pStream + streamSize;
	}
	static const std::byte* FromStream(const std::byte* pStream, glm::vec<N, T, Q>& data)
	{
		const auto streamSize = GetStreamSize(data);
		std::memcpy(&data, pStream, streamSize);
		return pStream + streamSize;
	}
};

template<typename T, int Rows, int Cols>
	requires (std::is_arithmetic_v<T>)
struct DBAdapter<Eigen::Matrix<T, Rows, Cols>>
{
	static size_t GetStreamSize(const Eigen::Matrix<T, Rows, Cols>& data)
	{
		return sizeof(Eigen::Matrix<T, Rows, Cols>);
	}
	static std::byte* ToStream(std::byte* pStream, const Eigen::Matrix<T, Rows, Cols>& data)
	{
		const auto streamSize = GetStreamSize(data);
		std::memcpy(pStream, &data, streamSize);
		return pStream + streamSize;
	}
	static const std::byte* FromStream(const std::byte* pStream, Eigen::Matrix<T, Rows, Cols>& data)
	{
		const auto streamSize = GetStreamSize(data);
		std::memcpy(&data, pStream, streamSize);
		return pStream + streamSize;
	}
};

template<typename T, size_t N>
	requires mcore::DBSerializable<T>
struct DBAdapter<T[N]>
{
	static size_t GetStreamSize(const T data[N])
	{
		auto size = 0;
		for (size_t i = 0; i < N; ++i)
		{
			size += DBAdapter<T>::GetStreamSize(data[i]);
		}
		return size;
	}
	static std::byte* ToStream(std::byte* pStream, const T data[N])
	{
		for (size_t i = 0; i < N; ++i)
		{
			pStream = DBAdapter<T>::ToStream(pStream, data[i]);
		}
		return pStream;
	}
	static const std::byte* FromStream(const std::byte* pStream, T data[N])
	{
		for (size_t i = 0; i < N; ++i)
		{
			pStream = DBAdapter<T>::FromStream(pStream, data[i]);
		}
		return pStream;
	}
};

template<typename T, size_t N>
	requires mcore::DBSerializable<T>
struct DBAdapter<std::array<T, N>>
{
	static size_t GetStreamSize(const std::array<T, N>& data)
	{
		size_t size = 0;
		for (const auto& x : data)
		{
			size += DBAdapter<T>::GetStreamSize(x);
		}
		return size;
	}
	static std::byte* ToStream(std::byte* pStream, const std::array<T, N>& data)
	{
		for (size_t i = 0; i < N; ++i)
		{
			pStream = DBAdapter<T>::ToStream(pStream, data[i]);
		}
		return pStream;
	}
	static const std::byte* FromStream(const std::byte* pStream, std::array<T, N>& data)
	{
		for (size_t i = 0; i < N; ++i)
		{
			pStream = DBAdapter<T>::FromStream(pStream, data[i]);
		}
		return pStream;
	}
};

template<typename T>
	requires mcore::DBSerializable<T>
struct DBAdapter<std::vector<T>>
{
	static size_t GetStreamSize(const std::vector<T>& data)
	{
		auto size = sizeof(size_t);
		for (const auto& x : data)
		{
			size += DBAdapter<T>::GetStreamSize(x);
		}
		return size;
	}
	static std::byte* ToStream(std::byte* pStream, const std::vector<T>& data)
	{
		const auto size = data.size();
		std::memcpy(pStream, &size, sizeof(size_t));
		pStream += sizeof(size_t);

		for (size_t i = 0; i < size; ++i)
		{
			pStream = DBAdapter<T>::ToStream(pStream, data[i]);
		}
		return pStream;
	}
	static const std::byte* FromStream(const std::byte* pStream, std::vector<T>& data)
	{
		size_t size = 0;
		std::memcpy(&size, pStream, sizeof(size_t));
		pStream += sizeof(size_t);

		data.resize(size);
		for (size_t i = 0; i < size; ++i)
		{
			pStream = DBAdapter<T>::FromStream(pStream, data[i]);
		}
		return pStream;
	}
};