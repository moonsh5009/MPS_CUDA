#pragma once
#include <type_traits>

#define ENUM_BIT_EXTENSION(NAME, FLAG_NAME) \
	using FLAG_NAME = mcore::EnumFlag<NAME>; \
	constexpr bool CheckFlags(FLAG_NAME flags, NAME bit) \
	{ \
		return (flags & bit) != 0; \
	} \
	constexpr FLAG_NAME operator|(NAME lhs, NAME rhs) \
	{ \
		return static_cast<std::underlying_type_t<NAME>>(lhs) | static_cast<std::underlying_type_t<NAME>>(rhs); \
	} \
	constexpr FLAG_NAME operator&(NAME lhs, NAME rhs) \
	{ \
		return static_cast<std::underlying_type_t<NAME>>(lhs) & static_cast<std::underlying_type_t<NAME>>(rhs); \
	} \
	constexpr FLAG_NAME operator~(NAME val) \
	{ \
		return ~static_cast<std::underlying_type_t<NAME>>(val); \
	}

namespace mcore
{
	template <typename T>
	concept IsEnum = std::is_enum_v<T>;

	template <typename T>
	concept EnumHasSize = std::is_enum_v<T> && requires {
		{ static_cast<std::underlying_type_t<T>>(T::Size) };
	};

	template <IsEnum E>
	constexpr int to_int(E e)
	{
		return static_cast<int>(e);
	}

	template <IsEnum E>
	constexpr unsigned to_uint(E e)
	{
		return static_cast<unsigned>(e);
	}

	template <IsEnum E>
	constexpr size_t to_sizet(E e)
	{
		return static_cast<size_t>(e);
	}

	template <IsEnum E>
	constexpr E from_int(int i)
	{
		return static_cast<E>(i);
	}

	template <IsEnum E>
	constexpr E from_uint(unsigned i)
	{
		return static_cast<E>(i);
	}

	template <IsEnum E>
	constexpr E from_sizet(size_t i)
	{
		return static_cast<E>(i);
	}

	template<EnumHasSize E>
	constexpr size_t enum_size()
	{
		return static_cast<size_t>(E::Size);
	}

	template<class ENUM>
		requires std::is_enum_v<ENUM>
	struct EnumFlag
	{
		using UnderlyingType = std::underlying_type_t<ENUM>;
		constexpr EnumFlag() : mask{ 0 } {}
		constexpr EnumFlag(UnderlyingType mask) : mask{ mask } {}
		constexpr EnumFlag(ENUM bit) : mask{ static_cast<UnderlyingType>(bit) } {}
		constexpr EnumFlag operator|(UnderlyingType mask) const
		{
			return this->mask | mask;
		}
		constexpr EnumFlag operator&(UnderlyingType mask) const
		{
			return this->mask & mask;
		}
		constexpr EnumFlag operator|(ENUM bit) const
		{
			return mask | static_cast<UnderlyingType>(bit);
		}
		constexpr EnumFlag operator&(ENUM bit) const
		{
			return mask & static_cast<UnderlyingType>(bit);
		}
		constexpr EnumFlag operator~() const
		{
			return ~mask;
		}
		template<typename N>
			requires std::is_integral_v<N>
		constexpr bool operator!=(N other) const
		{
			return mask != static_cast<UnderlyingType>(other);
		}
		constexpr bool operator!=(ENUM bit) const
		{
			return mask != static_cast<UnderlyingType>(bit);
		}
		constexpr operator UnderlyingType() const
		{
			return mask;
		}
		UnderlyingType mask;
	};
}