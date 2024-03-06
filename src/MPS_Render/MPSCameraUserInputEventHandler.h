#pragma once

#include "MPSCamera.h"

#include "HeaderPre.h"

namespace mps
{
	namespace rndr
	{
		class __MY_EXT_CLASS__ CameraUserInputEventHandler
		{
		public:
			CameraUserInputEventHandler() = delete;
			CameraUserInputEventHandler(Camera* pCamera);

		public:
			void OnWMouseDown(mevent::Flag flag, glm::ivec2 curPos);
			void OnWMouseUp(mevent::Flag flag, glm::ivec2 curPos);
			void OnMouseMove(mevent::Flag flag, glm::ivec2 curPos);
			void OnMouseWheel(mevent::Flag flag, glm::ivec2 offset, glm::ivec2 curPos);

		private:
			bool m_isDown;
			bool m_isRotate;
			glm::ivec2 m_startPos;

		private:
			Camera* m_pCamera;
		};
	}
}

#include "HeaderPost.h"

