#include "stdafx.h"
#include "DllLoader.h"

#include <filesystem>
#include <sstream>

namespace mcore::util
{
    DllLoader::~DllLoader()
    {
        Unload();
    }

    bool DllLoader::Load(const std::string& dllPath)
    {
        if (IsLoaded())
        {
            Unload();
        }

        std::filesystem::path absPath = std::filesystem::absolute(dllPath);

        m_hModule = LoadLibraryExA(absPath.string().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);

        if (!m_hModule)
        {
            DWORD error = GetLastError();
            return false;
        }

        m_dllPath = absPath.string();
        m_functionCache.clear();
        return true;
    }

    void DllLoader::Unload()
    {
        if (m_hModule)
        {
            FreeLibrary(m_hModule);
            m_hModule = nullptr;
            m_dllPath.clear();
            m_functionCache.clear();
        }
    }

    bool DllLoader::HasFunction(const std::string& functionName) const
    {
        if (!IsLoaded())
        {
            return false;
        }

        if (m_functionCache.find(functionName) != m_functionCache.end())
        {
            return true;
        }

        FARPROC procAddr = GetProcAddress(m_hModule, functionName.c_str());
        return procAddr != nullptr;
    }

    ScopedDllLoader::ScopedDllLoader(const std::string& dllPath)
    {
        if (!m_loader.Load(dllPath))
        {
            throw std::runtime_error("Failed to load DLL: " + dllPath);
        }
    }

    ScopedDllLoader::~ScopedDllLoader()
    {
        // m_loader의 소멸자가 자동으로 Unload 호출
    }

    bool ScopedDllLoader::HasFunction(const std::string& functionName) const
    {
        return m_loader.HasFunction(functionName);
    }

    DllManager& DllManager::Instance()
    {
        static DllManager instance;
        return instance;
    }

    std::shared_ptr<DllLoader> DllManager::LoadDll(const std::string& dllPath)
    {
        std::filesystem::path absPath = std::filesystem::absolute(dllPath);
        std::string key = absPath.string();

        auto it = m_dlls.find(key);
        if (it != m_dlls.end())
        {
            return it->second;
        }

        auto loader = std::make_shared<DllLoader>();
        if (!loader->Load(key))
        {
            return nullptr;
        }

        m_dlls[key] = loader;
        return loader;
    }

    std::shared_ptr<DllLoader> DllManager::GetDll(const std::string& dllPath)
    {
        std::filesystem::path absPath = std::filesystem::absolute(dllPath);
        std::string key = absPath.string();

        auto it = m_dlls.find(key);
        if (it != m_dlls.end())
        {
            return it->second;
        }

        return nullptr;
    }

    void DllManager::UnloadDll(const std::string& dllPath)
    {
        std::filesystem::path absPath = std::filesystem::absolute(dllPath);
        std::string key = absPath.string();

        auto it = m_dlls.find(key);
        if (it != m_dlls.end())
        {
            m_dlls.erase(it);
        }
    }

    void DllManager::UnloadAll()
    {
        m_dlls.clear();
    }
}