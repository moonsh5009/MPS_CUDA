#pragma once

#include <memory>
#include <functional>
#include <shared_mutex>

namespace mcore
{
    class ISignalBinding;
    class ISignal : public std::enable_shared_from_this<ISignal>
    {
    public:
        ISignal() = default;
        virtual ~ISignal() = default;
        ISignal(const ISignal&) = default;
        ISignal(ISignal&& other) = default;
        ISignal& operator=(const ISignal&) = default;
        ISignal& operator=(ISignal&&) = default;

        virtual void Disconnect(ISignalBinding*) = 0;
    };

    class ISignalBinding
    {
    public:
        ISignalBinding() = delete;
        ISignalBinding(const std::weak_ptr<ISignal>& pSignal)
            : m_pSignal{ pSignal }
        {}
        virtual ~ISignalBinding()
        {
            if (const auto pSignal = m_pSignal.lock())
            {
                pSignal->Disconnect(this);
            }
        }
        ISignalBinding(const ISignalBinding&) = default;
        ISignalBinding(ISignalBinding&& other) = default;
        ISignalBinding& operator=(const ISignalBinding&) = default;
        ISignalBinding& operator=(ISignalBinding&&) = default;

        bool Expired()
        {
            std::shared_lock lock{ mutex };
            return m_pSignal.expired();
        }
        void Reset()
        {
            std::unique_lock lock{ mutex };
            if (const auto pSignal = m_pSignal.lock())
            {
                pSignal->Disconnect(this);
                m_pSignal.reset();
            }
        }

    protected:
        std::shared_mutex mutex;
        std::weak_ptr<ISignal> m_pSignal;
    };

    template<typename Func>
    class SignalBinding : public ISignalBinding
    {
    public:
        SignalBinding(const std::weak_ptr<ISignal>& pSignal, std::function<Func>&& func)
            : ISignalBinding{ pSignal }, m_func{ std::move(func) }
        {}

        template<typename... Args>
            requires std::invocable<Func, Args...>
        void Invoke(Args... args)
        {
            std::shared_lock lock{ mutex };
            std::invoke(m_func, args...);
        }

    private:
        std::function<Func> m_func;
    };
}