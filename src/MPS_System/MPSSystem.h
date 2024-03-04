#pragma once

#include "../MPS_Render/MPSCameraUserInputEventHandler.h"

#include "MPSSystemEventHandler.h"

#include "HeaderPre.h"

namespace mgpu
{
	class MGPUCore;
}
namespace mps
{
	class GBArchiver;
	namespace rndr
	{
		class Camera;
		class RenderManager;
	}

	class Model;

	class __MY_EXT_CLASS__ System : public SystemEventHandler
	{
	public:
		System();

	public:
		void Initalize();

	protected:
		//virtual void OnLMouseDown(glm::ivec2 curPos, int mods) override;
		//virtual void OnRMouseDown(glm::ivec2 curPos, int mods) override;
		virtual void OnWMouseDown(glm::ivec2 curPos, int mods) override;
		//virtual void OnLMouseUp(glm::ivec2 curPos, int mods) override;
		//virtual void OnRMouseUp(glm::ivec2 curPos, int mods) override;
		virtual void OnWMouseUp(glm::ivec2 curPos, int mods) override;
		virtual void OnMouseMove(glm::ivec2 curPos) override;
		virtual void OnMouseWheel(glm::ivec2 curPos, glm::ivec2 offset, int mods) override;
		virtual void OnKeyDown(int key, int mods) override;
		virtual void OnKeyUp(int key, int mods) override;
		virtual void OnResize(int width, int height) override;
		virtual void OnUpdate() override;
		virtual void OnDraw() override;

	private:
		std::shared_ptr<mps::GBArchiver> m_pGBArchiver;

		std::shared_ptr<mps::rndr::RenderManager> m_pRenderManager;

		std::shared_ptr<mps::rndr::Camera> m_pCamera;
		std::unique_ptr<mps::rndr::CameraUserInputEventHandler> m_pCameraHandler;

	private:
		uint64_t m_frame;
		std::shared_ptr<mgpu::MGPUCore> m_pMGPUCore;

	private:
		void ResizeParticle(size_t size);
		std::shared_ptr<mps::Model> m_pSPHModel;
	};
}

#include "HeaderPost.h"