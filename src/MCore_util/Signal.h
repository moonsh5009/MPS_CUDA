#pragma once

#include "SignalBinding.h"
#include <vector>

namespace mcore
{
    class SignalSlot;
    template<typename Func>
    class SignalAgent : public ISignal
    {
    public:
        SignalAgent() = default;
        SignalAgent(size_t capacity) { m_bindings.reserve(capacity); };
        virtual ~SignalAgent() = default;
        SignalAgent(const SignalAgent&) = delete;
        SignalAgent(SignalAgent&& other) = default;
        SignalAgent& operator=(const SignalAgent&) = delete;
        SignalAgent& operator=(SignalAgent&&) = default;

        void Bind(const std::shared_ptr<ISignalBinding>& pBinding)
        {
            std::lock_guard<std::mutex> lock{ mutex };
            m_bindings.emplace_back(std::static_pointer_cast<SignalBinding<Func>>(pBinding));
        }

        template<typename... Args>
        requires std::invocable<Func, Args...>
        void Dispatch(Args... args)
        {
            std::lock_guard<std::mutex> lock{ mutex };
            for (auto weakBinding : m_bindings)
            {
                if(const auto pBinding = weakBinding.lock())
                    pBinding->Invoke(args...);
            }
        }

        void Disconnect(ISignalBinding* pBinding) override
        {
            /*std::lock_guard<std::mutex> lock{ mutex };
            auto it = std::remove(m_bindings.begin(), m_bindings.end(), pBinding);
            if (it != m_bindings.end())
                m_bindings.erase(it, m_bindings.end());*/
        }

    private:
        std::vector<std::weak_ptr<SignalBinding<Func>>> m_bindings;
        std::mutex mutex;
    };

    template<typename Func>
    using Signal = std::shared_ptr<SignalAgent<Func>>;

    template<typename Func>
    inline Signal<Func> MakeSignal()
    {
        return std::make_shared<mcore::SignalAgent<Func>>();
    }
}