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
		void Create();
		bool CompileShader(const Shader& pShader);
		void Destroy();
		
		void Bind() const;
		void Unbind() const;

	public:
		constexpr bool IsAssigned() const { return m_id != 0; }
		constexpr GLuint GetID() const { return m_id; }
		GLuint GetShader(const uint32_t i) const;
		GLuint GetVertexShader() const;
		GLuint GetFragmentShader() const;

	private:
		GLuint m_id;
		std::vector<GLuint> m_aAttachedShader;
	};
}

#include "HeaderPost.h"
