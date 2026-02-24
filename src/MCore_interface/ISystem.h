#pragma once

#include "IDBSession.h"
#include "IRenderCore.h"
#include "ISimulateManager.h"
#include "ISystemController.h"

#include <memory>

namespace mcore
{
	class ISystem
	{
	public:
		ISystem() = default;
		virtual ~ISystem() = default;
		ISystem(const ISystem&) = delete;
		ISystem(ISystem&&) = default;
		ISystem& operator=(const ISystem&) = delete;
		ISystem& operator=(ISystem&&) = default;

		virtual void Initialize(HWND window) = 0;
		virtual void Destroy() = 0;
		virtual void Run() = 0;

		IDBSession* GetDBSession() const { return m_pDBSession.get(); }
		IRenderCore* GetRenderCore() const { return m_pRenderCore.get(); }
		ISimulateManager* GetSimulateManager() const { return m_pSimulateManager.get(); }

		ISystemController* GetSystemController() const { return m_pSystemController.get(); }

		void SetSimulateManager(std::unique_ptr<ISimulateManager> pSimulateManager)
		{
			m_pSimulateManager = std::move(pSimulateManager);
		}

	protected:
		std::unique_ptr<IDBSession> m_pDBSession;
		std::unique_ptr<IRenderCore> m_pRenderCore;
		std::unique_ptr<ISimulateManager> m_pSimulateManager;

		std::unique_ptr<ISystemController> m_pSystemController;
	};
}