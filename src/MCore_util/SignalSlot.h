#pragma once

#include "Signal.h"
#include <assert.h>

namespace mcore
{
    class SignalSlot
    {
    public:
        SignalSlot() = default;
        virtual ~SignalSlot() = default;
        SignalSlot(const SignalSlot&) = delete;
        SignalSlot(SignalSlot&& other) = default;
        SignalSlot& operator=(const SignalSlot&) = delete;
        SignalSlot& operator=(SignalSlot&&) = default;

        template<typename Func, typename CallbackFunc>
        void Bind(const Signal<Func>& pSignal, CallbackFunc&& callback)
        {
            std::shared_ptr<ISignalBinding> pBinding = std::make_shared<SignalBinding<Func>>(pSignal, std::move(callback));
            pSignal->Bind(pBinding);
            m_pBindings.emplace_back(std::move(pBinding));
        }

        void Reset()
        {
            m_pBindings.clear();
        }

    private:
        std::vector<std::shared_ptr<ISignalBinding>> m_pBindings;
    };
}