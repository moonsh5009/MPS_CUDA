#pragma once

#include "MCUDAUtil.h"
#include <glm/gtc/quaternion.hpp>
#include <type_traits>

namespace mcore
{
    template<std::floating_point T>
    class MCUDA_HOST_DEVICE_FUNC Quaternion
    {
        using unit = glm::qua<T>;
        using unit_vec3 = glm::vec<3, T>;
        using unit_mat3 = glm::mat<3, 3, T>;
        using unit_mat4 = glm::mat<4, 4, T>;
    private:
        unit m_quaternion{ static_cast<T>(1.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.) };

    public:
        MCUDA_HOST_DEVICE_FUNC constexpr Quaternion() = default;
        MCUDA_HOST_DEVICE_FUNC constexpr Quaternion(const Quaternion&) = default;
        MCUDA_HOST_DEVICE_FUNC constexpr Quaternion(const unit& quaternion) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr Quaternion(const unit_vec3& eulerAngles) noexcept;
        MCUDA_HOST_DEVICE_FUNC Quaternion(T angle, const unit_vec3& axis) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr Quaternion(const unit_mat3& rotationMatrix) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr Quaternion(const unit_mat4& rotationMatrix) noexcept;

        MCUDA_HOST_DEVICE_FUNC constexpr void Set(const unit& quaternion) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void Set(const unit_vec3& eulerAngles) noexcept;
        MCUDA_HOST_DEVICE_FUNC void Set(T angle, const unit_vec3& axis) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void Set(const unit_mat3& rotationMatrix) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void Set(const unit_mat4& rotationMatrix) noexcept;

        MCUDA_HOST_DEVICE_FUNC constexpr void RotateGlobal(const unit_vec3& eulerAngles) noexcept;
        MCUDA_HOST_DEVICE_FUNC void RotateGlobal(T angle, const unit_vec3& axis) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void RotateLocal(const unit_vec3& eulerAngles) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void RotateFPS(T pitch, T yaw, const unit_vec3& referenceUp = { 0., 0., 1. }) noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr void Orient(const unit_vec3& forward, const unit_vec3& referenceUp = { 0., 0., 1. }) noexcept;
        
        MCUDA_HOST_DEVICE_FUNC constexpr void Normalize() noexcept;
        
        MCUDA_HOST_DEVICE_FUNC constexpr unit_vec3 GetEulerAngles() const noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr unit_mat4 GetMatrix() const noexcept;

        MCUDA_HOST_DEVICE_FUNC constexpr std::pair<T, unit_vec3> GetAngleAxis() const noexcept;
        MCUDA_HOST_DEVICE_FUNC constexpr Quaternion Inverse() const noexcept;

        MCUDA_HOST_DEVICE_FUNC constexpr operator const unit& () const noexcept { return m_quaternion; }
        MCUDA_HOST_DEVICE_FUNC constexpr Quaternion& operator*=(const Quaternion& rhs) noexcept
        {
            m_quaternion *= rhs.m_quaternion;
            return *this;
        }
        MCUDA_HOST_DEVICE_FUNC constexpr bool operator!=(const Quaternion& rhs) noexcept
        {
            return m_quaternion != rhs.m_quaternion;
        }

        static MCUDA_HOST_DEVICE_FUNC constexpr Quaternion Slerp(const Quaternion& lhs, const Quaternion& rhs, T weight) noexcept;
        static MCUDA_HOST_DEVICE_FUNC Quaternion SlerpLongest(const Quaternion& lhs, const Quaternion& rhs, T weight) noexcept;
    };

    using Quaternionf = Quaternion<float>;
}

#include "Quaternion.inl"