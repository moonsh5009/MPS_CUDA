#pragma once

#include "../MCore_interface/ISystem.h"

#include "HeaderPre.h"

namespace mcore::system
{
	class __MY_EXT_CLASS__ System : public ISystem
	{
	public:
		~System() override;

		void Initialize(HWND window) override;
		void Destroy() override;
		void Run() override;
	};
}

#include "HeaderPost.h"