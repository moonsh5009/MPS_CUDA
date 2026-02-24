#include "stdafx.h"
#include "SimulateManager.h"

#include "DeviceContainerFactory.h"
#include "SimulateStepFactory.h"

using namespace mcore;

simulate::SimulateManager::~SimulateManager()
{
	for (auto stream : m_streams)
	{
		if (stream)
			cudaStreamDestroy(stream);
	}
}

void simulate::SimulateManager::Initialize()
{
	m_deviceContainers = simulate::DeviceContainerFactory::Instance().Build(this);
	m_stepGroups = simulate::SimulateStepFactory::Instance().Build(this);
}

void simulate::SimulateManager::Simulate()
{
	for (auto& group : m_stepGroups)
	{
		const size_t count = group.size();
		if (count == 0) continue;

		if (count == 1)
		{
			group[0]->Execute(nullptr, m_dt);
		}
		else
		{
			EnsureStreams(count);
			for (size_t i = 0; i < count; ++i)
			{
				group[i]->Execute(m_streams[i], m_dt);
			}
			for (size_t i = 0; i < count; ++i)
			{
				cudaStreamSynchronize(m_streams[i]);
			}
		}
	}
}

void simulate::SimulateManager::EnsureStreams(size_t count)
{
	while (m_streams.size() < count)
	{
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		m_streams.push_back(stream);
	}
}
