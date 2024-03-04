#pragma once

#include <glm/glm.hpp>

#include "HeaderPre.h"

namespace mps
{
	namespace rndr
	{
		class __MY_EXT_CLASS__ CameraProjection
		{
		public:
			CameraProjection();

		public:
			constexpr float GetHeight() const noexcept { return m_height; }
			constexpr float GetAspectRatio() const noexcept { return m_aspectRatio; }
			constexpr float GetZNear() const noexcept { return m_zNear; }
			constexpr float GetZFar() const noexcept { return m_zFar; }

			constexpr void SetHeight(float height) noexcept;
			constexpr void SetAspectRatio(float aspectRatio) noexcept;
			constexpr void SetAspectRatio(int width, int height) noexcept;
			constexpr void SetZNear(float zNear) noexcept;
			constexpr void SetZFar(float zFar) noexcept;

		public:
			constexpr glm::fmat4 GetMatrix() noexcept { return m_projMat; }
			constexpr glm::fmat4 GetInvMatrix() noexcept { return m_projInvMat; }
			bool UpdateMatrix();

		private:
			bool m_isDirty;
			bool m_isOrtho;

			float m_height;
			float m_aspectRatio;

			float m_zNear;
			float m_zFar;

			glm::fmat4 m_projMat;
			glm::fmat4 m_projInvMat;
		};
	}
}

#include "HeaderPost.h"
