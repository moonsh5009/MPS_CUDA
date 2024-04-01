#pragma once

#include <cstdint>
#include <vector>

#include "GL/GLEW/glew.h"
#include "GL/GLEW/wglew.h"

namespace mgl
{
	template<typename T, GLenum TYPE = GL_DYNAMIC_DRAW>
	class Buffer
	{
	public:
		Buffer() : m_id{ 0u}, m_size{ 0ull }, m_capacity{ 0ull }
		{
			Create();
		}
		Buffer(const size_t size) : m_id{ 0u }, m_size{ 0ull }, m_capacity{ 0ull }
		{
			Resize(size);
		}
		Buffer(const std::vector<T>& host) : m_id{ 0u }, m_size{ 0ull }, m_capacity{ 0ull }
		{
			Resize(host);
		}
		Buffer(const Buffer&) = delete;
		Buffer(Buffer&& other)
		{
			*this = std::move(other);
		}
		Buffer& operator=(const Buffer&) = delete;
		Buffer& operator=(Buffer&& other)
		{
			if (this != &other)
			{
				m_id = other.m_id;
				m_size = other.m_size;
				m_capacity = other.m_capacity;
				other.m_id = 0u;
				other.m_size = 0ull;
				other.m_capacity = 0ull;
			}
			return *this;
		}
		virtual ~Buffer()
		{
			Destroy();
		}

	public:
		constexpr bool IsAssigned() const { return m_id != 0; }
		constexpr GLuint GetID() const { return m_id; }
		constexpr size_t GetSize() const { return m_size; }
		constexpr size_t GetCapacity() const { return m_capacity; }
		constexpr size_t GetStride() const { return sizeof(T); }
		constexpr size_t GetByteLength() const { return m_size * sizeof(T); }
		constexpr size_t GetCapacityByteLength() const { return m_capacity * sizeof(T); }

	public:
		void Create();
		void Clear();
		virtual void Destroy();

		virtual void Resize(const size_t size);
		virtual void Resize(const std::vector<T>& host);
		virtual void ShrinkToFit();

		void CopyFromDevice(const Buffer<T>& src);
		void CopyFromDevice(const Buffer<T>& src, const size_t dstOffset);
		void CopyFromDevice(const Buffer<T>& src, const size_t dstOffset, const size_t srcOffset, const size_t byteLength);

		void CopyFromHost(const std::vector<T>& src);
		void CopyFromHost(const std::vector<T>& src, const size_t dstOffset);
		void CopyFromHost(const std::vector<T>& src, const size_t dstOffset, const size_t srcOffset, const size_t byteLength);

		template<typename T2>
		void CopyFromHost(const T2& src);
		template<typename T2>
		void CopyFromHost(const T2& src, const size_t dstOffset);
		template<typename T2>
		void CopyFromHost(const T2& src, const size_t dstOffset, const size_t srcOffset, const size_t byteLength);

	protected:
		GLuint m_id;
		size_t m_size;
		size_t m_capacity;
	};
}

#include "MGLBuffer.inl"