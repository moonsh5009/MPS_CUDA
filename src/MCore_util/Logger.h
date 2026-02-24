#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <concepts>
#include <array>
#ifdef _WIN32
#include <Windows.h>
#endif

#include "HeaderPre.h"

namespace mcore
{
    class __MY_EXT_CLASS__ Logger
    {
    public:
        enum class Level
        {
            Info,
            Warning,
            Error,
            Debug,
        };

        static void Print();
        static void Info(const std::string& message)
        {
            Log(Level::Info, message);
        }
        static void Warning(const std::string& message)
        {
            Log(Level::Warning, message);
        }
        static void Error(const std::string& message)
        {
            Log(Level::Error, message);
        }
        static void Debug(const std::string& message)
        {
            Log(Level::Debug, message);
        }
        static void InfoNoNewLine(const std::string& message)
        {
            LogNoNewLine(Level::Info, message);
        }
        static void WarningNoNewLine(const std::string& message)
        {
            LogNoNewLine(Level::Warning, message);
        }
        static void ErrorNoNewLine(const std::string& message)
        {
            LogNoNewLine(Level::Error, message);
        }
        static void DebugNoNewLine(const std::string& message)
        {
            LogNoNewLine(Level::Debug, message);
        }

        template<typename... Args>
            requires (sizeof...(Args) > 1 || (!std::same_as<std::decay_t<Args>, std::string> && ...))
        static void Info(Args&&... args)
        {
            std::ostringstream oss;
            (oss << ... << args);
            Info(oss.str());
        }
        template<typename... Args>
            requires (sizeof...(Args) > 1 || (!std::same_as<std::decay_t<Args>, std::string> && ...))
        static void Warning(Args&&... args)
        {
            std::ostringstream oss;
            (oss << ... << args);
            Warning(oss.str());
        }
        template<typename... Args>
            requires (sizeof...(Args) > 1 || (!std::same_as<std::decay_t<Args>, std::string> && ...))
        static void Error(Args&&... args)
        {
            std::ostringstream oss;
            (oss << ... << args);
            Error(oss.str());
        }
        template<typename... Args>
            requires (sizeof...(Args) > 1 || (!std::same_as<std::decay_t<Args>, std::string> && ...))
        static void Debug(Args&&... args)
        {
            std::ostringstream oss;
            (oss << ... << args);
            Debug(oss.str());
        }

        template<typename... Args>
            requires (sizeof...(Args) > 1 || (!std::same_as<std::decay_t<Args>, std::string> && ...))
        static void InfoNoNewLine(Args&&... args)
        {
            std::ostringstream oss;
            (oss << ... << args);
            InfoNoNewLine(oss.str());
        }
        template<typename... Args>
            requires (sizeof...(Args) > 1 || (!std::same_as<std::decay_t<Args>, std::string> && ...))
        static void WarningNoNewLine(Args&&... args)
        {
            std::ostringstream oss;
            (oss << ... << args);
            WarningNoNewLine(oss.str());
        }
        template<typename... Args>
            requires (sizeof...(Args) > 1 || (!std::same_as<std::decay_t<Args>, std::string> && ...))
        static void ErrorNoNewLine(Args&&... args)
        {
            std::ostringstream oss;
            (oss << ... << args);
            ErrorNoNewLine(oss.str());
        }
        template<typename... Args>
            requires (sizeof...(Args) > 1 || (!std::same_as<std::decay_t<Args>, std::string> && ...))
        static void DebugNoNewLine(Args&&... args)
        {
            std::ostringstream oss;
            (oss << ... << args);
            DebugNoNewLine(oss.str());
        }

    private:
        static void Log(Level level, const std::string& message);
        static void LogNoNewLine(Level level, const std::string& message);
        static std::string m_message;
    };
}

#include "HeaderPost.h"