#pragma once

#include <glm/glm.hpp>

#include "HeaderPre.h"

namespace mps
{
	namespace rndr
	{
		class __MY_EXT_CLASS__ CameraTransform
		{
		public:
			CameraTransform();

		public:
			void SetPosition(const glm::fvec3& pos) { m_transMat[3] = { pos, 1.0f }; m_isDirty = false; }
			void SetXDir(const glm::fvec3& xDir) { m_rotateMat[0] = { xDir, 0.0f }; m_isDirty = false; }
			void SetYDir(const glm::fvec3& yDir) { m_rotateMat[1] = { yDir, 0.0f }; m_isDirty = false; }
			void SetZDir(const glm::fvec3& zDir) { m_rotateMat[2] = { -zDir, 0.0f }; m_isDirty = false; }

			glm::fvec3 GetPosition() const { return m_transMat[3]; }
			glm::fvec3 GetXDir() const { return m_rotateMat[0]; }
			glm::fvec3 GetYDir() const { return m_rotateMat[1]; }
			glm::fvec3 GetZDir() const { return -m_rotateMat[2]; }

			void Rotate(float x, float y, float radian);

		public:
			constexpr glm::fmat4 GetMatrix() noexcept { return m_viewMat; }
			constexpr glm::fmat4 GetInvMatrix() noexcept { return m_viewInvMat; }
			bool UpdateMatrix();

		private:
			bool m_isDirty;

			glm::fvec3 m_position;

			glm::fmat4 m_transMat;
			glm::fmat4 m_rotateMat;
			glm::fmat4 m_viewMat;

			glm::fmat4 m_viewInvMat;
		};
	}
}

#include "HeaderPost.h"
