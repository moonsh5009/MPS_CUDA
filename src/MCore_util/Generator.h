#pragma once

#include <coroutine>
#include <optional>

namespace mcore
{
	namespace util
	{
        template<typename T>
        struct Generator
        {
            struct promise_type
            {
                std::optional<T> current_value;

                Generator get_return_object()
                {
                    return Generator{
                        std::coroutine_handle<promise_type>::from_promise(*this)
                    };
                }
                std::suspend_always initial_suspend() { return {}; }
                std::suspend_always final_suspend() noexcept { return {}; }
                std::suspend_always yield_value(T value)
                {
                    current_value = std::move(value);
                    return {};
                }
                void unhandled_exception()
                {
                    std::exit(1);
                }
                void return_void() {}
            };

            using handle_type = std::coroutine_handle<promise_type>;

            explicit Generator(handle_type h) : handle(h) {}
            Generator(const Generator&) = delete;
            Generator& operator=(const Generator&) = delete;
            Generator(Generator&& other) noexcept : handle(other.handle)
            {
                other.handle = nullptr;
            }
            ~Generator()
            {
                if (handle)
                    handle.destroy();
            }

            T next()
            {
                if (!handle.done())
                {
                    handle.resume();
                }
                return *handle.promise().current_value;
            }

            T value() const
            {
                return *handle.promise().current_value;
            }

        private:
            handle_type handle;
        };
	}
}