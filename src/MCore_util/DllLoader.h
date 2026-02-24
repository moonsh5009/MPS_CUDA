#pragma once

#include <windows.h>
#include <string>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <functional>

#include "HeaderPre.h"

namespace mcore::util
{
    class __MY_EXT_CLASS__ DllLoader
    {
    public:
        DllLoader() = default;
        ~DllLoader();
        DllLoader(const DllLoader&) = delete;
        DllLoader(DllLoader&&) noexcept = default;
        DllLoader& operator=(const DllLoader&) = delete;
        DllLoader& operator=(DllLoader&&) noexcept = default;

        bool Load(const std::string& dllPath);
        void Unload();

        template<typename FuncType>
        FuncType GetFunction(const std::string& functionName);

        bool HasFunction(const std::string& functionName) const;

        bool IsLoaded() const { return m_hModule != nullptr; }

        std::string GetPath() const { return m_dllPath; }

    private:
        HMODULE m_hModule = nullptr;
        std::string m_dllPath;
        std::unordered_map<std::string, FARPROC> m_functionCache;
    };

    // RAII 스타일 DLL 로더
    class __MY_EXT_CLASS__ ScopedDllLoader
    {
    public:
        explicit ScopedDllLoader(const std::string& dllPath);
        ~ScopedDllLoader();
        ScopedDllLoader(const ScopedDllLoader&) = delete;
        ScopedDllLoader(ScopedDllLoader&&) noexcept = default;
        ScopedDllLoader& operator=(const ScopedDllLoader&) = delete;
        ScopedDllLoader& operator=(ScopedDllLoader&&) noexcept = default;

        template<typename FuncType>
        FuncType GetFunction(const std::string& functionName);

        bool HasFunction(const std::string& functionName) const;
        bool IsLoaded() const { return m_loader.IsLoaded(); }

    private:
        DllLoader m_loader;
    };

    // DLL 관리자 (싱글톤)
    class __MY_EXT_CLASS__ DllManager
    {
    public:
        static DllManager& Instance();

        std::shared_ptr<DllLoader> LoadDll(const std::string& dllPath);
        std::shared_ptr<DllLoader> GetDll(const std::string& dllPath);
        void UnloadDll(const std::string& dllPath);
        void UnloadAll();

    private:
        DllManager() = default;
        ~DllManager() = default;
        DllManager(const DllManager&) = delete;
        DllManager& operator=(const DllManager&) = delete;

        std::unordered_map<std::string, std::shared_ptr<DllLoader>> m_dlls;
    };

    template<typename FuncType>
    FuncType DllLoader::GetFunction(const std::string& functionName)
    {
        if (!IsLoaded())
        {
            throw std::runtime_error("DLL is not loaded");
        }

        auto it = m_functionCache.find(functionName);
        if (it != m_functionCache.end())
        {
            return reinterpret_cast<FuncType>(it->second);
        }

        FARPROC procAddr = GetProcAddress(m_hModule, functionName.c_str());
        if (!procAddr)
        {
            throw std::runtime_error("Function not found: " + functionName);
        }

        m_functionCache[functionName] = procAddr;
        return reinterpret_cast<FuncType>(procAddr);
    }

    template<typename FuncType>
    FuncType ScopedDllLoader::GetFunction(const std::string& functionName)
    {
        return m_loader.GetFunction<FuncType>(functionName);
    }
}

#include "HeaderPost.h"