#pragma once

#include "MCUDAUtil.h"
#include <glm/glm.hpp>
#include <type_traits>
#include <climits>

namespace mcore
{
    template<std::floating_point T>
    class AABB final
    {
    public:
        using unit = glm::vec<3, T>;
        static constexpr auto num_max = std::numeric_limits<T>::max();

        MCUDA_HOST_DEVICE_FUNC constexpr AABB();
        MCUDA_HOST_DEVICE_FUNC constexpr AABB(const unit& vMin, const unit& vMax);
		MCUDA_HOST_DEVICE_FUNC constexpr ~AABB() = default;
        MCUDA_HOST_DEVICE_FUNC constexpr AABB(const AABB& target)
        {
			*this = target;
        }
		MCUDA_HOST_DEVICE_FUNC AABB(AABB&&) = default;
        MCUDA_HOST_DEVICE_FUNC constexpr AABB& operator=(const AABB& target) noexcept
        {
            if (this == &target)
				return *this;
            m_min = target.m_min;
            m_max = target.m_max;
            return *this;
        }
		MCUDA_HOST_DEVICE_FUNC AABB& operator=(AABB&&) = default;

        MCUDA_HOST_DEVICE_FUNC constexpr void Initialize() noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void Set(const unit& vMin, const unit& vMax) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void Scale(T scaleFactor) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void Scale(const unit& centerPoint, T scaleFactor) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void Union(const AABB& target) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void Intersect(const AABB& target) noexcept;

        MCUDA_HOST_DEVICE_FUNC constexpr float GetSurfaceArea() const noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr unit GetCenter() const noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr float GetMinimumDistanceFromPlane(glm::vec4& pvPlane) const;

        MCUDA_HOST_DEVICE_FUNC constexpr bool IsAvailable() const noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr bool IsInside(const unit& pos) const;
        MCUDA_HOST_DEVICE_FUNC constexpr bool IsInside(const AABB& aabb) const;
        MCUDA_HOST_DEVICE_FUNC constexpr bool IsIntersect(const AABB& aabb) const;

        MCUDA_HOST_DEVICE_FUNC constexpr AABB operator+(const AABB& target)
        {
            return { glm::min(m_min, target.m_min), glm::max(m_max, target.m_max) };
        }

        MCUDA_HOST_DEVICE_FUNC constexpr AABB& operator+=(const AABB& target)
        {
            m_min = glm::min(m_min, target.m_min);
            m_max = glm::max(m_max, target.m_max);
            return *this;
        }

        MCUDA_HOST_DEVICE_FUNC constexpr AABB operator+(const unit& target)
        {
            return { glm::min(m_min, target), glm::max(m_max, target) };
        }

        MCUDA_HOST_DEVICE_FUNC constexpr AABB& operator+=(const unit& target)
        {
            m_min = glm::min(m_min, target);
            m_max = glm::max(m_max, target);
            return *this;
        }

        MCUDA_HOST_DEVICE_FUNC  constexpr bool operator==(const AABB& target) const noexcept
        {
            return m_max == target.m_max && m_min == target.m_min;
        }

        MCUDA_HOST_DEVICE_FUNC  constexpr bool operator!=(const AABB& target) const noexcept
        {
            return !(this->operator==(target));
        }

        MCUDA_HOST_DEVICE_FUNC constexpr unit& GetMin() noexcept { return m_min; }
        MCUDA_HOST_DEVICE_FUNC constexpr unit& GetMax() noexcept { return m_max; }
        MCUDA_HOST_DEVICE_FUNC constexpr const unit& GetMin() const noexcept { return m_min; }
        MCUDA_HOST_DEVICE_FUNC constexpr const unit& GetMax() const noexcept { return m_max; }

    private:
        unit m_min;
        unit m_max;
    };

    using AABBf = AABB<float>;
}

#include "AABB.inl"