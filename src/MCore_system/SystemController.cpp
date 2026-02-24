#include "stdafx.h"
#include "SystemController.h"

#include "../MCore_interface/ISystem.h"
#include "../MCore_util/VulkanCore.h"

#include "SimulateToRenderConverterFactory.h"

using namespace mcore::system;

SystemController::SystemController(ISystem* pSystem)
	: ISystemController{ pSystem }
{}

void SystemController::Initialize()
{
	m_simulateToRenderModelConverters = SimulateToRenderConverterFactory::Instance().Build();

	Bind(GetSystem()->GetDBSession()->commitSignal, [&](const DBNotifyInfoArray& infos)
	{
		cudaDeviceSynchronize();
		mvk::VulkanCore::Instance()->WaitIdle();

		for (const auto& info : infos)
		{
			switch (info.updateType)
			{
			case DBUpdateType::ADD:
				Add(info);
				break;
			case DBUpdateType::MODIFY:
				Modify(info);
				break;
			case DBUpdateType::REMOVE:
				Delete(info);
				break;
			default:
				break;
			}
		}
		for (const auto& group : GetSystem()->GetSimulateManager()->GetStepGroups())
		{
			for (const auto& pStep : group)
			{
				pStep->OnUpdatedContainer();
			}
		}
		SimulateToRender();
	});
}

void SystemController::SimulateToRender() const
{
	for (const auto& pConverter : m_simulateToRenderModelConverters)
	{
		if (pConverter)
		{
			pConverter->Convert(GetSystem()->GetSimulateManager(), GetSystem()->GetRenderCore()->GetModelContainer());
		}
	}
	GetSystem()->GetRenderCore()->Invalidate();
}

void SystemController::Add(const DBNotifyInfo& info) const
{
	auto pDeviceContainer = GetSystem()->GetSimulateManager()->GetDeviceContainer(info.typeId);
	if (!pDeviceContainer)
	{
		throw std::runtime_error("Failed to generate simulation object container.");
	}
	pDeviceContainer->AddDB(info.pData.get());
}

void SystemController::Modify(const DBNotifyInfo& info) const
{
	auto pDeviceContainer = GetSystem()->GetSimulateManager()->GetDeviceContainer(info.typeId);
	if (!pDeviceContainer)
	{
		throw std::runtime_error("Failed to generate simulation object.");
	}
	pDeviceContainer->ModifyDB(info.pPrevData.get(), info.pData.get());
}

void SystemController::Delete(const DBNotifyInfo& info) const
{
	auto pDeviceContainer = GetSystem()->GetSimulateManager()->GetDeviceContainer(info.typeId);
	if (!pDeviceContainer)
	{
		throw std::runtime_error("Failed to generate simulation object.");
	}
	pDeviceContainer->DeleteDB(info.key, info.pPrevData.get());
}
