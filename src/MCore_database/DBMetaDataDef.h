#pragma once

#include "../MCore_interface/DBDef.h"
#include "../MCore_database/Data.h"
#include "../MCore_database/Ref.h"

using namespace mcore;

#define REGISTRY_DATABASE(NAME, ID, HASH_SIZE) \
	struct DBMetaData##NAME \
	{ \
		static constexpr auto name = #NAME; \
		static constexpr mcore::DBTypeID id = ID; \
		static constexpr size_t hash_size = HASH_SIZE; \
	};

#define REGISTRY_SINGLE_DATABASE(NAME, ID) \
	struct DBMetaData##NAME \
	{ \
		static constexpr auto name = #NAME; \
		static constexpr mcore::DBTypeID id = ID; \
	};