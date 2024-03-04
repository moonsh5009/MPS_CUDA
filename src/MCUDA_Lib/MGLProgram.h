#pragma once

#include "HeaderPre.h"

namespace mgl
{
	class Shader;
	class __MY_EXT_CLASS__ Program
	{
	public:
		Program();
		~Program();
		Program(const Program&) = delete;
		Program(Program&& other)
		{
			*this = std::move(other);
		}
		Program& operator=(const Program&) = delete;
		Program& operator=(Program&& other)
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

	public:
		void Create();
		bool CompileShader(const Shader& pShader);
		void Destroy();
		void Bind() const;
		void Unbind() const;

	private:
		GLuint m_id;
	};
}

#include "HeaderPost.h"
