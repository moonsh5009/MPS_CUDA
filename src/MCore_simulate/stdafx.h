#pragma once
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)

#define BUILD_MCORE_SIMULATE
#define MCORE_USE_CUDA

#define NOMINMAX
#define NOGDI

#include <iostream>
#include <cassert>
#include <cstdint>

#include <algorithm>
#include <array>
#include <stack>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <string>
#include <tuple>
#include <optional>
#include <functional>

#include "../MCore_util/DeviceSingleArray.h"
#include "../MCore_util/DeviceMultiArray.h"