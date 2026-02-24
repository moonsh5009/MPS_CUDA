#pragma once

#include "../MCore_util/TypeDef.h"

namespace mcore
{
	using DBKey = unsigned long long;
	constexpr DBKey NullKey = 0;

	using DBTypeID = unsigned long long;
	constexpr DBTypeID TYPE_ID_MAX = 2048;

	enum class ErrorType
	{
		LOST_DB_POOL,
		SIZE,
	};
}