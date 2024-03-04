#pragma once

#include "MGLBuffer.h"

#include "HeaderPre.h"

namespace mgl
{
	class __MY_EXT_CLASS__ VertexArray
	{
	public:
		VertexArray();
		~VertexArray();

	public:
		constexpr bool IsAssigned() const noexcept { return m_id != 0; }

	public:
		void Create();
		void Delete();
		void Bind();
		void Unbind();

	public:
		template<typename T>
		constexpr std::tuple<GLint, GLenum> GetBufferType()
		{
			if constexpr (std::is_same_v<T, int>)
				return std::make_tuple(1, GL_INT);
			else if constexpr (std::is_same_v<T, glm::ivec2>)
				return std::make_tuple(2, GL_INT);
			else if constexpr (std::is_same_v<T, glm::ivec3>)
				return std::make_tuple(3, GL_INT);
			else if constexpr (std::is_same_v<T, glm::ivec4>)
				return std::make_tuple(4, GL_INT);

			else if constexpr (std::is_same_v<T, unsigned>)
				return std::make_tuple(1, GL_UNSIGNED_INT);
			else if constexpr (std::is_same_v<T, glm::uvec2>)
				return std::make_tuple(2, GL_UNSIGNED_INT);
			else if constexpr (std::is_same_v<T, glm::uvec3>)
				return std::make_tuple(3, GL_UNSIGNED_INT);
			else if constexpr (std::is_same_v<T, glm::uvec4>)
				return std::make_tuple(4, GL_UNSIGNED_INT);

			else if constexpr (std::is_same_v<T, float>)
				return std::make_tuple(1, GL_FLOAT);
			else if constexpr (std::is_same_v<T, glm::fvec2>)
				return std::make_tuple(2, GL_FLOAT);
			else if constexpr (std::is_same_v<T, glm::fvec3>)
				return std::make_tuple(3, GL_FLOAT);
			else if constexpr (std::is_same_v<T, glm::fvec4>)
				return std::make_tuple(4, GL_FLOAT);
			else if constexpr (std::is_same_v<T, glm::fmat4>)
				return std::make_tuple(16, GL_FLOAT);

			else if constexpr (std::is_same_v<T, double>)
				return std::make_tuple(1, GL_DOUBLE);
			else if constexpr (std::is_same_v<T, glm::dvec2>)
				return std::make_tuple(2, GL_DOUBLE);
			else if constexpr (std::is_same_v<T, glm::dvec3>)
				return std::make_tuple(3, GL_DOUBLE);
			else if constexpr (std::is_same_v<T, glm::dvec4>)
				return std::make_tuple(4, GL_DOUBLE);
			else if constexpr (std::is_same_v<T, glm::dmat4>)
				return std::make_tuple(16, GL_DOUBLE);

			else static_assert("Invalid Type");
		}

		template<typename T>
		void AddVertexBuffer(const mgl::Buffer<T>& vbo)
		{
			if (!IsAssigned()) return;

			const auto[size, type] = GetBufferType<T>();

			glVertexArrayVertexBuffer(m_id, m_index, vbo.GetID(), 0, sizeof(T));
			glEnableVertexArrayAttrib(m_id, m_index);
			glVertexArrayAttribBinding(m_id, m_index, m_index);
			glVertexArrayAttribFormat(m_id, m_index, size, type, GL_FALSE, 0);
			m_index++;
		}

	private:
		GLuint m_id;
		GLuint m_index;
	};
}

#include "HeaderPost.h"
