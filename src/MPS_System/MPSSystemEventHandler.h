#pragma once

#include "../MCUDA_Lib/MEventPack.h"

#include "HeaderPre.h"

namespace mgpu
{
	class MGPUCore;
}
namespace mps
{
	class __MY_EXT_CLASS__ SystemEventHandler
	{
	public:
		SystemEventHandler();

	protected:
		void Initalize();

	public:
		void LMouseDown(mevent::Flag flag, glm::ivec2 curPos);
		void RMouseDown(mevent::Flag flag, glm::ivec2 curPos);
		void WMouseDown(mevent::Flag flag, glm::ivec2 curPos);
		void LMouseUp(mevent::Flag flag, glm::ivec2 curPos);
		void RMouseUp(mevent::Flag flag, glm::ivec2 curPos);
		void WMouseUp(mevent::Flag flag, glm::ivec2 curPos);
		void MouseMove(mevent::Flag flag, glm::ivec2 curPos);
		void MouseWheel(mevent::Flag flag, glm::ivec2 offset, glm::ivec2 curPos);
		void KeyDown(uint32_t key, uint32_t repCnt, mevent::Flag flag);
		void KeyUp(uint32_t key, uint32_t repCnt, mevent::Flag flag);
		void Resize(int width, int height);
		void Update();
		void Draw();

	protected:
		virtual void OnLMouseDown(mevent::Flag flag, glm::ivec2 curPos);
		virtual void OnRMouseDown(mevent::Flag flag, glm::ivec2 curPos);
		virtual void OnWMouseDown(mevent::Flag flag, glm::ivec2 curPos);
		virtual void OnLMouseUp(mevent::Flag flag, glm::ivec2 curPos);
		virtual void OnRMouseUp(mevent::Flag flag, glm::ivec2 curPos);
		virtual void OnWMouseUp(mevent::Flag flag, glm::ivec2 curPos);
		virtual void OnMouseMove(mevent::Flag flag, glm::ivec2 curPos);
		virtual void OnMouseWheel(mevent::Flag flag, glm::ivec2 offset, glm::ivec2 curPos);
		virtual void OnKeyDown(uint32_t key, uint32_t repCnt, mevent::Flag flag);
		virtual void OnKeyUp(uint32_t key, uint32_t repCnt, mevent::Flag flag);
		virtual void OnResize(int width, int height);
		virtual void OnUpdate() = 0;
		virtual void OnDraw() = 0;
	};
}

#include "HeaderPost.h"