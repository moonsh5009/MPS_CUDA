#pragma once

#include "../MCore_util/SignalSlot.h"

#include "../MCore_interface/DBNotifyInfo.h"
#include "../MCore_interface/ISystemController.h"
#include "../MCore_interface/ISimulateToRenderConverter.h"

#include "HeaderPre.h"

namespace mcore::system
{
	class __MY_EXT_CLASS__ SystemController : public ISystemController, public SignalSlot
	{
	public:
		SystemController(ISystem* pSystem);

		void Initialize() override;
		void SimulateToRender() const override;

	private:
		void Add(const DBNotifyInfo& info) const;
		void Modify(const DBNotifyInfo& info) const;
		void Delete(const DBNotifyInfo& info) const;

	protected:
		SimulateToRenderConverterArray m_simulateToRenderModelConverters;
	};
}

#include "HeaderPost.h"