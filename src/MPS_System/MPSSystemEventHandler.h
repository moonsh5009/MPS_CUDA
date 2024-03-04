#pragma once

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
		void LMouseDown(glm::ivec2 curPos, int mods);
		void RMouseDown(glm::ivec2 curPos, int mods);
		void WMouseDown(glm::ivec2 curPos, int mods);
		void LMouseUp(glm::ivec2 curPos, int mods);
		void RMouseUp(glm::ivec2 curPos, int mods);
		void WMouseUp(glm::ivec2 curPos, int mods);
		void MouseMove(glm::ivec2 curPos);
		void MouseWheel(glm::ivec2 curPos, glm::ivec2 offset, int mods);
		void KeyDown(int key, int mods);
		void KeyUp(int key, int mods);
		void Resize(int width, int height);
		void Update();
		void Draw();

	protected:
		virtual void OnLMouseDown(glm::ivec2 curPos, int mods);
		virtual void OnRMouseDown(glm::ivec2 curPos, int mods);
		virtual void OnWMouseDown(glm::ivec2 curPos, int mods);
		virtual void OnLMouseUp(glm::ivec2 curPos, int mods);
		virtual void OnRMouseUp(glm::ivec2 curPos, int mods);
		virtual void OnWMouseUp(glm::ivec2 curPos, int mods);
		virtual void OnMouseMove(glm::ivec2 curPos);
		virtual void OnMouseWheel(glm::ivec2 curPos, glm::ivec2 offset, int mods);
		virtual void OnKeyDown(int key, int mods);
		virtual void OnKeyUp(int key, int mods);
		virtual void OnResize(int width, int height);
		virtual void OnUpdate() = 0;
		virtual void OnDraw() = 0;
	};
}

#include "HeaderPost.h"