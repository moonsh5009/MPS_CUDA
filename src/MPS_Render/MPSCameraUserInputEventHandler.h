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
			void OnWMouseDown(glm::ivec2 curPos, int mods);
			void OnWMouseUp(glm::ivec2 curPos, int mods);
			void OnMouseMove(glm::ivec2 curPos);
			void OnMouseWheel(glm::ivec2 curPos, glm::ivec2 offset, int mods);

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

