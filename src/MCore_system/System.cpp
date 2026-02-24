#include "stdafx.h"
#include "System.h"

#include "../MCore_util/VulkanCore.h"

#include "../MCore_database/Session.h"
#include "../MCore_render/RenderCore.h"
#include "../MCore_simulate/SimulateManager.h"

#include "SystemController.h"

using namespace mcore::system;

mcore::system::System::~System()
{
	Destroy();
}

void System::Initialize(HWND window)
{
	// Initialize CUDA
	CUDA_CHECK(cudaDeviceReset());
	CUDA_CHECK(cudaSetDevice(0));
	int n;
	cudaGetDeviceCount(&n);
	OutputDebugStringA(std::to_string(n).c_str());

	// Initialize Vulkan
	mvk::VulkanCore::Initialize(window);

	m_pDBSession = std::make_unique<database::Session>();
	m_pDBSession->Initialize();

	if (!m_pSimulateManager)
	{
		m_pSimulateManager = std::make_unique<simulate::SimulateManager>();
	}
	m_pSimulateManager->Initialize();

	m_pRenderCore = std::make_unique<render::RenderCore>();
	m_pRenderCore->Initialize();

	m_pSystemController = std::make_unique<SystemController>(this);
	m_pSystemController->Initialize();
}

void System::Destroy()
{
	cudaDeviceSynchronize();
	mvk::VulkanCore::Instance()->WaitIdle();

	m_pSystemController.reset();
	m_pRenderCore.reset();
	m_pSimulateManager.reset();
	m_pDBSession.reset();

	mvk::VulkanCore::ShutDown();
}

void System::Run()
{
	m_pSimulateManager->Simulate();
	m_pRenderCore->Invalidate();
}
