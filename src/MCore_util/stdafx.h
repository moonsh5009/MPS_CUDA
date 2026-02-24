#pragma once
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)

#if defined __NVCC__
#pragma warning (disable : 4068)
#endif

#define BUILD_MCORE_UTIL
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

#include "Logger.h"