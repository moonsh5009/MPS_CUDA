#include "Quaternion.h"

template<std::floating_point T>
constexpr mcore::Quaternion<T>::Quaternion(const unit& quaternion) noexcept :
    m_quaternion{ quaternion }
{
}

template<std::floating_point T>
constexpr mcore::Quaternion<T>::Quaternion(const unit_vec3& eulerAngles) noexcept :
    m_quaternion{ eulerAngles }
{}

template<std::floating_point T>
mcore::Quaternion<T>::Quaternion(T angle, const unit_vec3& axis) noexcept :
    m_quaternion{ glm::angleAxis(angle, glm::normalize(axis)) }
{}

template<std::floating_point T>
constexpr mcore::Quaternion<T>::Quaternion(const unit_mat3& rotationMatrix) noexcept :
    m_quaternion{ rotationMatrix }
{}
template<std::floating_point T>
constexpr mcore::Quaternion<T>::Quaternion(const unit_mat4& rotationMatrix) noexcept :
    m_quaternion{ rotationMatrix }
{}

template<std::floating_point T>
constexpr void mcore::Quaternion<T>::Set(const unit& quaternion) noexcept
{
    m_quaternion = quaternion;
}

template<std::floating_point T>
constexpr void mcore::Quaternion<T>::Set(const unit_vec3& eulerAngles) noexcept
{
    m_quaternion = { eulerAngles };
}

template<std::floating_point T>
void mcore::Quaternion<T>::Set(T angle, const unit_vec3& axis) noexcept
{
    m_quaternion = glm::angleAxis(angle, glm::normalize(axis));
}

template<std::floating_point T>
constexpr void mcore::Quaternion<T>::Set(const unit_mat3& rotationMatrix) noexcept
{
    m_quaternion = { rotationMatrix };
}

template<std::floating_point T>
constexpr void mcore::Quaternion<T>::Set(const unit_mat4& rotationMatrix) noexcept
{
    m_quaternion = { rotationMatrix };
}

template<std::floating_point T>
constexpr void mcore::Quaternion<T>::RotateGlobal(const unit_vec3& eulerAngles) noexcept
{
    m_quaternion = unit{ eulerAngles } * m_quaternion;
}

template<std::floating_point T>
void mcore::Quaternion<T>::RotateGlobal(T angle, const unit_vec3& axis) noexcept
{
    m_quaternion = glm::angleAxis(angle, glm::normalize(axis)) * m_quaternion;
}

template<std::floating_point T>
constexpr void mcore::Quaternion<T>::RotateLocal(const unit_vec3& eulerAngles) noexcept
{
    const auto matrix = GetMatrix();
    RotateGlobal(eulerAngles[0], matrix[0]);
    RotateGlobal(eulerAngles[1], matrix[1]);
    RotateGlobal(eulerAngles[2], matrix[2]);
}

template<std::floating_point T>
constexpr void mcore::Quaternion<T>::RotateFPS(T pitch, T yaw, const unit_vec3& referenceUp) noexcept
{
    const auto matrix = GetMatrix();
    const auto normRefUp = glm::normalize(referenceUp);
    const auto horiz = unit_vec3{ matrix[0] };
    const auto refHoriz = horiz - glm::dot(normRefUp, horiz) * normRefUp;
    RotateGlobal(pitch, refHoriz);
    RotateGlobal(yaw, normRefUp);
}

template<std::floating_point T>
constexpr void mcore::Quaternion<T>::Orient(const unit_vec3& forward, const unit_vec3& referenceUp) noexcept
{
    const auto normForward = glm::normalize(forward);
    const auto normRefUp = glm::normalize(referenceUp);

    const auto horizontal = glm::normalize(glm::cross(normRefUp, normForward));
    const auto vertical = glm::normalize(glm::cross(normForward, horizontal));

    m_quaternion = unit_mat3{ horizontal, vertical, normForward };
    m_quaternion = glm::normalize(m_quaternion);
}

template<std::floating_point T>
constexpr void mcore::Quaternion<T>::Normalize() noexcept
{
    m_quaternion = glm::normalize(m_quaternion);
}

template<std::floating_point T>
constexpr mcore::Quaternion<T>::unit_vec3 mcore::Quaternion<T>::GetEulerAngles() const noexcept
{
    return glm::eulerAngles(m_quaternion);
}

template<std::floating_point T>
constexpr mcore::Quaternion<T>::unit_mat4 mcore::Quaternion<T>::GetMatrix() const noexcept
{
    return glm::mat4_cast(m_quaternion);
}

template<std::floating_point T>
constexpr std::pair<T, typename mcore::Quaternion<T>::unit_vec3> mcore::Quaternion<T>::GetAngleAxis() const noexcept
{
    return { glm::angle(m_quaternion), glm::axis(m_quaternion) };
}

template<std::floating_point T>
constexpr mcore::Quaternion<T>mcore::Quaternion<T>::Inverse() const noexcept
{
    return glm::inverse(m_quaternion);
}

template<std::floating_point T>
constexpr mcore::Quaternion<T> mcore::Quaternion<T>::Slerp(const Quaternion& lhs, const Quaternion& rhs, T weight) noexcept
{
    return glm::slerp(lhs.m_quaternion, rhs.m_quaternion, weight);
}

template<std::floating_point T>
inline MCUDA_HOST_DEVICE_FUNC mcore::Quaternion<T> mcore::Quaternion<T>::SlerpLongest(const Quaternion& lhs, const Quaternion& rhs, T weight) noexcept
{
    const auto& lQuat = lhs.m_quaternion;
    const auto& rQuat = rhs.m_quaternion;

    auto cosTheta = glm::dot(lQuat, rQuat);
    if (cosTheta > (static_cast<T>(1.) - glm::epsilon<T>()))
        return glm::lerp(lQuat, rQuat, weight);

    unit rQuat_adj;
    if (cosTheta >= static_cast<T>(0.))
    {
        rQuat_adj = -rQuat;
        cosTheta = -cosTheta;
    }
    else
        rQuat_adj = rQuat;

    const auto theta = acos(cosTheta);
    return (((sin((static_cast<T>(1.) - weight) * theta) * lQuat) + sin(weight * theta) * rQuat_adj) / sin(theta));
}
