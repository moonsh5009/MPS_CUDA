#include <array>
#include <glm/gtc/type_ptr.hpp>
#include "AABB.h"

template<std::floating_point T>
constexpr mcore::AABB<T>::AABB()
    : m_min{ num_max, num_max, num_max }
    , m_max{ -num_max, -num_max, -num_max }
{}

template<std::floating_point T>
constexpr mcore::AABB<T>::AABB(const unit& vMin, const unit& vMax)
    : m_min{ vMin }
    , m_max{ vMax }
{}

template<std::floating_point T>
constexpr void mcore::AABB<T>::Initialize() noexcept
{
    m_min = { num_max, num_max, num_max };
    m_max = { -num_max, -num_max, -num_max };
}

template<std::floating_point T>
constexpr void mcore::AABB<T>::Set(const unit& vMin, const unit& vMax) noexcept
{
    m_min = vMin;
    m_max = vMax;
}

template<std::floating_point T>
constexpr void mcore::AABB<T>::Scale(T scaleFactor) noexcept
{
    Scale(GetCenter(), scaleFactor);
}

template<std::floating_point T>
constexpr void mcore::AABB<T>::Scale(const unit& centerPoint, T scaleFactor) noexcept
{
    m_min = (m_min - centerPoint) * scaleFactor + centerPoint;
    m_max = (m_max - centerPoint) * scaleFactor + centerPoint;
}

template<std::floating_point T>
constexpr void mcore::AABB<T>::Union(const AABB& target) noexcept
{
    if (m_min.x > target.m_min.x) m_min.x = target.m_min.x;
    if (m_min.y > target.m_min.y) m_min.y = target.m_min.y;
    if (m_min.z > target.m_min.z) m_min.z = target.m_min.z;
    if (m_max.x < target.m_max.x) m_max.x = target.m_max.x;
    if (m_max.y < target.m_max.y) m_max.y = target.m_max.y;
    if (m_max.z < target.m_max.z) m_max.z = target.m_max.z;
}

template<std::floating_point T>
constexpr void mcore::AABB<T>::Intersect(const AABB& target) noexcept
{
    if (m_min.x < target.m_min.x) m_min.x = target.m_min.x;
    if (m_min.y < target.m_min.y) m_min.y = target.m_min.y;
    if (m_min.z < target.m_min.z) m_min.z = target.m_min.z;
    if (m_max.x > target.m_max.x) m_max.x = target.m_max.x;
    if (m_max.y > target.m_max.y) m_max.y = target.m_max.y;
    if (m_max.z > target.m_max.z) m_max.z = target.m_max.z;
}

template<std::floating_point T>
constexpr float mcore::AABB<T>::GetSurfaceArea() const noexcept
{
    const auto size = m_max - m_min;
    const auto half = size.x * size.y + size.y * size.z + size.z * size.x;
    return half + half;
}

template<std::floating_point T>
constexpr mcore::AABB<T>::unit mcore::AABB<T>::AABB::GetCenter() const noexcept
{
    return (m_min + m_max) * static_cast<T>(0.5);
}

template<std::floating_point T>
constexpr float mcore::AABB<T>::GetMinimumDistanceFromPlane(glm::vec4& plane) const
{
    return (plane.x > 0 ? m_min.x : m_max.x) * plane.x
        + (plane.y > 0 ? m_min.x : m_max.y) * plane.y
        + (plane.z > 0 ? m_min.x : m_max.z) * plane.z +
        + plane.w;
}

template<std::floating_point T>
constexpr bool mcore::AABB<T>::AABB::IsAvailable() const noexcept
{
    return m_min.x <= m_max.x
        && m_min.y <= m_max.y
        && m_min.z <= m_max.z;
}

template<std::floating_point T>
constexpr bool mcore::AABB<T>::IsInside(const unit& pos) const
{
    return pos.x < m_max.x
        && pos.y < m_max.y
        && pos.z < m_max.z
        && pos.x > m_min.x
        && pos.y > m_min.y
        && pos.z > m_min.z;
}

template<std::floating_point T>
constexpr bool mcore::AABB<T>::IsInside(const AABB& aabb) const
{
    if (!IsAvailable() || !aabb.IsAvailable())
        return false;

    return aabb.m_max.x < m_max.x
        && aabb.m_max.y < m_max.y
        && aabb.m_max.z < m_max.z
        && aabb.m_min.x > m_min.x
        && aabb.m_min.y > m_min.y
        && aabb.m_min.z > m_min.z;
}

template<std::floating_point T>
constexpr bool mcore::AABB<T>::IsIntersect(const AABB& aabb) const
{
    if (!IsAvailable() || !aabb.IsAvailable())
        return false;

    return aabb.m_min.x < m_max.x
        && aabb.m_min.y < m_max.y
        && aabb.m_min.z < m_max.z
        && aabb.m_max.x > m_min.x
        && aabb.m_max.y > m_min.y
        && aabb.m_max.z > m_min.z;
}