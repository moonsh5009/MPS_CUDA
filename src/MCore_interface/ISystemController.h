#pragma once

namespace mcore
{
	class ISystem;
	class ISystemController
	{
	public:
		ISystemController(ISystem* pSystem)
			: m_pSystem{ pSystem }
		{}
		virtual ~ISystemController() = default;
		ISystemController(const ISystemController&) = delete;
		ISystemController(ISystemController&&) = default;
		ISystemController& operator=(const ISystemController&) = delete;
		ISystemController& operator=(ISystemController&&) = default;

		virtual void Initialize() = 0;
		virtual void SimulateToRender() const = 0;

		ISystem* GetSystem() const { return m_pSystem; }

	protected:
		ISystem* m_pSystem;
	};
}