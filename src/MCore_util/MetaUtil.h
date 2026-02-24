#pragma once

#include <algorithm>
#include <type_traits>
#include <array>
#include <string_view>

namespace mcore
{
    namespace meta
    {
        template<typename Type, Type N>
        constexpr Type MaxValue()
        {
            return N;
        }

        template<typename Type, Type N, Type M, Type... Rest>
        constexpr Type MaxValue()
        {
            constexpr Type tail = MaxValue<Type, M, Rest...>();
            return N > tail ? N : tail;
        }

        consteval char ToUpperChar(char c)
        {
            return (c >= 'a' && c <= 'z') ? (c - ('a' - 'A')) : c;
        }

        template <std::size_t N>
        consteval std::array<char, N> ToUpperCharArray(const char(&str)[N])
        {
            std::array<char, N> tmp{};
            for (std::size_t i = 0; i < N; i++)
            {
                tmp[i] = ToUpperChar(str[i]);
            }
            return tmp;
        }

        template <std::size_t N>
        consteval std::string_view ToUpper(const std::array<char, N>& arr)
        {
            return std::string_view(arr.data(), N);
        }
    }
}