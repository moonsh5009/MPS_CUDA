#pragma once

#include "HeaderPre.h"

namespace mgl
{
	class __MY_EXT_CLASS__ Shader
	{
	public:
		Shader();
		Shader(const std::string_view& srcPath, GLenum kind);
		~Shader();
		Shader(const Shader&) = delete;
		Shader(Shader&& other)
		{
			*this = std::move(other);
		}
		Shader& operator=(const Shader&) = delete;
		Shader& operator=(Shader&& other)
		{
			if (this != &other)
			{
				m_id = other.m_id;
				other.m_id = 0u;
			}
			return *this;
		}

	public:
		constexpr GLuint GetID() const { return m_id; }
		constexpr bool IsAssigned() const { return m_id != 0; }

		void Create(const std::string_view& srcPath, GLenum kind);
		void Destroy();

	private:
		std::string Load(const std::string_view& srcPath);

	private:
		GLuint m_id;
	};
}

#include "HeaderPost.h"