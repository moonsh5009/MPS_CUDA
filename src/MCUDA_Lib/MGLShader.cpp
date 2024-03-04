#include "stdafx.h"
#include "MGLShader.h"

#include <fstream>
#include <sstream>
#include <regex>

mgl::Shader::Shader() : m_id{ 0u }
{
}

mgl::Shader::Shader(const std::string_view& srcPath, GLenum kind)
{
	Create(srcPath, kind);
}

mgl::Shader::~Shader()
{
	Destroy();
}

void mgl::Shader::Create(const std::string_view& srcPath, GLenum kind)
{
	Destroy();

	const auto src = Load(srcPath);
	const auto srcPtr = src.data();

	m_id = glCreateShader(kind);
	glShaderSource(m_id, 1, &srcPtr, NULL);
	glCompileShader(m_id);
}

void mgl::Shader::Destroy()
{
	if (!IsAssigned()) return;

	glDeleteShader(m_id);
	m_id = 0;
}

std::string mgl::Shader::Load(const std::string_view& srcPath)
{
	std::string path = "../shader/";
	path += srcPath;

	std::string source;

	const std::regex dirPattern{ "\\w+(\\.glsl|\\.vert|\\.frag)" };
	const std::regex includePattern{ "#include\\s*<(.*?)>", std::regex::optimize };

	const std::function<void(const std::string&)> ReadSource = [&](const std::string& path)
	{
		const std::string pathDir = std::regex_replace(path, dirPattern, "");

		std::stringstream ss;
		std::ifstream srcFile(path);
		srcFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		ss << srcFile.rdbuf();
		srcFile.close();

		std::string line;
		std::smatch matches;
		while (std::getline(ss, line))
		{
			if (std::regex_search(line, matches, includePattern))
				ReadSource(pathDir + matches[1].str());
			else
				source.append(line + '\n');
		}
	};

	try
	{
		ReadSource(path);
	}
	catch (std::ifstream::failure e)
	{
		assert("Unable to open shader source File");
		return {};
	}
	return source;
}