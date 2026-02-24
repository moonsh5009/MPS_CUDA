#include "stdafx.h"
#include "Logger.h"

using namespace mcore;

std::string Logger::m_message = "";

void mcore::Logger::Print()
{
    std::cout << m_message << std::endl;

#ifdef _WIN32
    OutputDebugStringA((m_message + "\n").c_str());
#endif

    m_message = "";
}

void mcore::Logger::Log(Level level, const std::string& message)
{
    std::string prefix;
    /*switch (level)
    {
    case Level::Info:    prefix = "[INFO] "; break;
    case Level::Warning: prefix = "[WARN] "; break;
    case Level::Error:   prefix = "[ERROR] "; break;
    case Level::Debug:   prefix = "[DEBUG] "; break;
    }*/

    std::string fullMessage = prefix + message;

    assert(level != Level::Error);
    m_message += fullMessage + "\n";
}

void mcore::Logger::LogNoNewLine(Level level, const std::string& message)
{
    std::string prefix;
    /*switch (level)
    {
    case Level::Info:    prefix = "[INFO] "; break;
    case Level::Warning: prefix = "[WARN] "; break;
    case Level::Error:   prefix = "[ERROR] "; break;
    case Level::Debug:   prefix = "[DEBUG] "; break;
    }*/

    std::string fullMessage = prefix + message;

    assert(level != Level::Error);
    m_message += fullMessage;
}
