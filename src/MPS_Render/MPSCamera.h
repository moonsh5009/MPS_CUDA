#pragma once

#include "MPSCameraTransform.h"
#include "MPSCameraProjection.h"

#include "../MCUDA_Lib/MEventPack.h"
#include "../MPS_Object/MPSUniformDef.h"

#include "HeaderPre.h"

typedef mgpu::MEventPack<std::function<void()>> MCameraUpdateEventListener;

namespace mps
{
	class GBArchiver;
	namespace rndr
	{
		class __MY_EXT_CLASS__ Camera
		{
		public:
			Camera();
			void Update(mps::GBArchiver* pGBArchiver);

		private:
			bool UpdateMatrix();

		public:
			void MoveForward(const float x);
			void Move(const glm::fvec2 dif);

		public:
			CameraTransform* GetTransform() { return m_pTransform.get(); }
			CameraProjection* GetProjection() { return m_pProjection.get(); }
			MCameraUpdateEventListener& GetUpdateListener() { return m_updateListener; }

		private:
			std::unique_ptr<CameraTransform> m_pTransform;
			std::unique_ptr<CameraProjection> m_pProjection;
			mps::CameraParam m_param;

		private:
			std::weak_ptr<GBArchiver> m_pGBArchiver;

		private:
			MCameraUpdateEventListener m_updateListener;
		};
	}
}

#include "HeaderPost.h"