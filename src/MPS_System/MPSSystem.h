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
		void SetDevice(int device);

	protected:
		//virtual void OnLMouseDown(mevent::Flag flag, glm::ivec2 curPos) override;
		//virtual void OnRMouseDown(mevent::Flag flag, glm::ivec2 curPos) override;
		virtual void OnWMouseDown(mevent::Flag flag, glm::ivec2 curPos) override;
		//virtual void OnLMouseUp(mevent::Flag flag, glm::ivec2 curPos) override;
		//virtual void OnRMouseUp(mevent::Flag flag, glm::ivec2 curPos) override;
		virtual void OnWMouseUp(mevent::Flag flag, glm::ivec2 curPos) override;
		virtual void OnMouseMove(mevent::Flag flag, glm::ivec2 curPos) override;
		virtual void OnMouseWheel(mevent::Flag flag, glm::ivec2 offset, glm::ivec2 curPos) override;
		virtual void OnKeyDown(uint32_t key, uint32_t repCnt, mevent::Flag flag) override;
		virtual void OnKeyUp(uint32_t key, uint32_t repCnt, mevent::Flag flag) override;
		virtual void OnResize(int width, int height) override;
		virtual void OnUpdate() override;
		virtual void OnDraw() override;

		void Capture(uint32_t endFrame, uint32_t step);

	private:
		std::shared_ptr<mps::GBArchiver> m_pGBArchiver;

		std::shared_ptr<mps::rndr::RenderManager> m_pRenderManager;

		std::shared_ptr<mps::rndr::Camera> m_pCamera;
		std::unique_ptr<mps::rndr::CameraUserInputEventHandler> m_pCameraHandler;

	private:
		bool b_runSim;
		uint64_t m_frame;
		std::shared_ptr<mgpu::MGPUCore> m_pMGPUCore;

	private:
		uint32_t m_particleWSize;
		void ResizeParticle();
		void ViscosityTestScene(size_t height);
		void SurfaceTensionTestScene();
		std::shared_ptr<mps::Model> m_pSPHModel;
		std::shared_ptr<mps::Model> m_pBoundaryModel;
	};
}

#include "HeaderPost.h"