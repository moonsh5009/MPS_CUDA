#pragma once
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)

#if defined __NVCC__
#pragma warning (disable : 4068)
#endif

#define BUILD_MCORE_RENDER
#define MCORE_USE_CUDA

#define NOMINMAX
#define NOGDI

#include <Windows.h>
#include <iostream>
#include <cassert>
#include <cstdint>

#include <algorithm>
#include <array>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <string>
#include <tuple>
#include <optional>
#include <functional>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "../MCore_util/EnumExt.h"
#include "../MCore_util/Logger.h"