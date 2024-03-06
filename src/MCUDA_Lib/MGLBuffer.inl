#include "MGLBuffer.h"
template<typename T>
void mgl::Buffer<T>::Create()
{
	if (IsAssigned()) return;

	GLuint tmp;
	glCreateBuffers(1, &tmp);
	m_id = tmp;
}

template<typename T>
void mgl::Buffer<T>::Destroy()
{
	if (!IsAssigned()) return;

	glDeleteBuffers(1, &m_id);
	m_id = 0u;
	m_size = 0ull;
	m_capacity = 0ull;
}

template<typename T>
void mgl::Buffer<T>::Clear()
{
	if (!IsAssigned()) return;

	m_size = 0ull;
}

template<typename T>
void mgl::Buffer<T>::Resize(const size_t size)
{
	Create();

	m_size = size;
	if (m_capacity >= m_size) return;

	m_capacity = m_size;
	glNamedBufferData(m_id, GetByteLength(), nullptr, GL_DYNAMIC_DRAW);
}

template<typename T>
void mgl::Buffer<T>::Resize(const std::vector<T>& host)
{
	Create();

	m_size = host.size();
	if (m_capacity >= m_size)
	{
		glNamedBufferSubData(m_id, 0, GetByteLength(), reinterpret_cast<const void*>(host.data()));
		return;
	}

	m_capacity = m_size;
	glNamedBufferData(m_id, GetByteLength(), reinterpret_cast<const void*>(host.data()), GL_DYNAMIC_DRAW);
}

template<typename T>
void mgl::Buffer<T>::ShrinkToFit()
{
	if (m_size == m_capacity) return;

	if (m_size == 0ull)
	{
		Destroy();
		return;
	}

	m_capacity = m_size;
	glNamedBufferData(m_id, GetByteLength(), nullptr, GL_DYNAMIC_DRAW);
}

template<typename T>
void mgl::Buffer<T>::CopyFromDevice(const Buffer<T>& src)
{
	if (m_size < src.GetSize()) { assert(false); return; }

	glNamedCopyBufferSubDataEXT(src.GetID(), m_id, 0, 0, src.GetByteLength());
}

template<typename T>
void mgl::Buffer<T>::CopyFromDevice(const Buffer<T>& src, const size_t dstOffset)
{
	const auto dstByteLength = GetByteLength();
	if (dstByteLength < dstOffset) { assert(false); return; }

	const auto srcByteLength = src.size() * sizeof(T);
	if (dstByteLength - dstOffset < srcByteLength) { assert(false); return; }

	glNamedCopyBufferSubDataEXT(src.GetID(), m_id, 0, dstOffset, srcByteLength);
}

template<typename T>
void mgl::Buffer<T>::CopyFromDevice(const Buffer<T>& src, const size_t dstOffset, const size_t srcOffset, const size_t byteLength)
{
	const auto dstByteLength = GetByteLength();
	if (dstByteLength < dstOffset) { assert(false); return; }

	const auto srcByteLength = src.size() * sizeof(T);
	if (srcByteLength < srcOffset) { assert(false); return; }

	if (srcByteLength - srcOffset < byteLength) { assert(false); return; }
	if (dstByteLength - dstOffset < byteLength) { assert(false); return; }

	glNamedCopyBufferSubDataEXT(src.GetID(), m_id, srcOffset, dstOffset, byteLength);
}

template<typename T>
void mgl::Buffer<T>::CopyFromHost(const std::vector<T>& src)
{
	if (m_size < src.size()) { assert(false); return; }

	glNamedBufferSubData(m_id, 0, GetByteLength(), reinterpret_cast<const char*>(src.data()));
}

template<typename T>
void mgl::Buffer<T>::CopyFromHost(const std::vector<T>& src, const size_t dstOffset)
{
	const auto dstByteLength = GetByteLength();
	if (dstByteLength < dstOffset) { assert(false); return; }

	const auto srcByteLength = src.size() * sizeof(T);
	if (dstByteLength - dstOffset < srcByteLength) { assert(false); return; }

	glNamedBufferSubData(m_id, dstOffset, srcByteLength, reinterpret_cast<const char*>(src.data()));
}

template<typename T>
void mgl::Buffer<T>::CopyFromHost(const std::vector<T>& src, const size_t dstOffset, const size_t srcOffset, const size_t byteLength)
{
	const auto dstByteLength = GetByteLength();
	if (dstByteLength < dstOffset) { assert(false); return; }

	const auto srcByteLength = src.size() * sizeof(T);
	if (srcByteLength < srcOffset) { assert(false); return; }

	if (srcByteLength - srcOffset < byteLength) { assert(false); return; }
	if (dstByteLength - dstOffset < byteLength) { assert(false); return; }

	glNamedBufferSubData(m_id, dstOffset, byteLength, reinterpret_cast<const char*>(src.data()) + srcOffset);
}

template<typename T> template<typename T2>
void mgl::Buffer<T>::CopyFromHost(const T2& src)
{
	const auto dstByteLength = GetByteLength();
	const auto srcByteLength = sizeof(T2);
	if (dstByteLength < srcByteLength) { assert(false); return; }

	glNamedBufferSubData(m_id, 0, srcByteLength, reinterpret_cast<const char*>(&src));
}

template<typename T> template<typename T2>
void mgl::Buffer<T>::CopyFromHost(const T2& src, const size_t dstOffset)
{
	const auto dstByteLength = GetByteLength();
	const auto srcByteLength = sizeof(T2);
	if (dstByteLength < dstOffset + srcByteLength) { assert(false); return; }

	glNamedBufferSubData(m_id, dstOffset, srcByteLength, reinterpret_cast<const char*>(&src));
}

template<typename T> template<typename T2>
void mgl::Buffer<T>::CopyFromHost(const T2& src, const size_t dstOffset, const size_t srcOffset, const size_t byteLength)
{
	const auto dstByteLength = GetByteLength();
	if (dstByteLength < dstOffset) { assert(false); return; }

	const auto srcByteLength = sizeof(T);
	if (srcByteLength < srcOffset) { assert(false); return; }

	if (srcByteLength - srcOffset < byteLength) { assert(false); return; }
	if (dstByteLength - dstOffset < byteLength) { assert(false); return; }

	glNamedBufferSubData(m_id, dstOffset, byteLength, reinterpret_cast<const char*>(&src) + srcOffset);
}
