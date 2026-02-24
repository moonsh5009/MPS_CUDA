#include "stdafx.h"
#include "Commander.h"

#include "VulkanCore.h"
#include "RenderContext.h"

using namespace mcore;

namespace
{
    constexpr std::array<std::tuple<mvk::CommandType, vk::CommandPoolCreateFlags>, 3> TYPE_TO_FLAGBITS = {
        std::tuple{ mvk::CommandType::PREBAKED, vk::CommandPoolCreateFlags{} },
        std::tuple{ mvk::CommandType::REUSABLE, vk::CommandPoolCreateFlagBits::eResetCommandBuffer },
        std::tuple{ mvk::CommandType::DYNAMIC, vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient },
    };
}

mvk::CommandID::CommandID(const std::shared_ptr<Commander>& pCommander, CommandIDData data)
    : m_pCommander{ pCommander }, m_data{ data }
{
    ++pCommander->m_commandIDCounts[to_sizet(data.queueType)][to_sizet(data.commandType)][data.index];
}

mvk::CommandID::~CommandID()
{
    Destroy();
}

mvk::CommandID& mvk::CommandID::operator=(const CommandID& other)
{
    if (this != &other)
    {
        if (const auto pCommander = other.m_pCommander.lock(); pCommander)
        {
            m_pCommander = other.m_pCommander;
            m_data = other.m_data;
            ++pCommander->m_commandIDCounts[to_sizet(m_data.queueType)][to_sizet(m_data.commandType)][m_data.index];
        }
    }
    return *this;
}

void mvk::CommandID::Destroy()
{
    if (const auto pCommander = m_pCommander.lock(); pCommander)
    {
        if (--pCommander->m_commandIDCounts[to_sizet(m_data.queueType)][to_sizet(m_data.commandType)][m_data.index] == 0)
        {
            pCommander->m_idleCommandBuffers.emplace_back(m_data.queueType, m_data.commandType, m_data.index);
        }
        m_pCommander.reset();
    }
}

mvk::SemaphoreID::SemaphoreID(const std::shared_ptr<Commander>& pCommander, size_t index)
    : m_pCommander{ pCommander }, m_index{ index }
{
    ++pCommander->m_semaphoreIDCounts[index];
}

mvk::SemaphoreID::~SemaphoreID()
{
    Destroy();
}

mvk::SemaphoreID& mvk::SemaphoreID::operator=(const SemaphoreID& other)
{
    if (this != &other)
    {
        if (const auto pCommander = other.m_pCommander.lock(); pCommander)
        {
            m_pCommander = other.m_pCommander;
            m_index = other.m_index;
            ++pCommander->m_semaphoreIDCounts[m_index];
        }
    }
    return *this;
}

void mvk::SemaphoreID::Destroy()
{
    if (const auto pCommander = m_pCommander.lock(); pCommander)
    {
        if (--pCommander->m_semaphoreIDCounts[m_index] == 0)
        {
            pCommander->m_idleSemaphores.emplace_back(m_index);
        }
        m_pCommander.reset();
    }
}

mvk::FenceID::FenceID(const std::shared_ptr<Commander>& pCommander, size_t index)
    : m_pCommander{ pCommander }, m_index{ index }
{
    ++pCommander->m_fenceIDCounts[index];
}

mvk::FenceID::~FenceID()
{
    Destroy();
}

mvk::FenceID& mvk::FenceID::operator=(const FenceID& other)
{
    if (this != &other)
    {
        if (const auto pCommander = other.m_pCommander.lock(); pCommander)
        {
            m_pCommander = other.m_pCommander;
            m_index = other.m_index;
            ++pCommander->m_fenceIDCounts[m_index];
        }
    }
    return *this;
}

void mvk::FenceID::Destroy()
{
    if (const auto pCommander = m_pCommander.lock(); pCommander)
    {
        if (--pCommander->m_fenceIDCounts[m_index] == 0)
        {
            pCommander->m_idleFences.emplace_back(m_index);
        }
        m_pCommander.reset();
    }
}

mvk::CommandQueue::CommandQueue(const std::weak_ptr<Commander>& pCommander)
    : m_pCommander{ pCommander }
{
    m_commandQueues[0].reserve(16);
    m_waitSemaphores.reserve(16);
}

void mvk::CommandQueue::Reset()
{
    m_tempCommandbuffers.clear();
}

void mvk::CommandQueue::AddSemaphore(const SemaphoreID& semaphoreID, vk::PipelineStageFlags2 stageMask)
{
    if (m_pCommander.expired()) return;

    m_waitSemaphores.emplace_back(semaphoreID, stageMask);
}

void mvk::CommandQueue::AddCommand(const CommandID& commandID)
{
    const auto pCommander = m_pCommander.lock();
    if (!pCommander) return;

    m_commandQueues[to_sizet(pCommander->GetQueueType(commandID))]
        .emplace_back(commandID);
}

void mvk::CommandQueue::AddCommand(CommandID&& commandID)
{
    const auto pCommander = m_pCommander.lock();
    if (!pCommander) return;

    m_commandQueues[to_sizet(pCommander->GetQueueType(commandID))]
        .emplace_back(std::move(commandID));
}

void mvk::CommandQueue::AddCommandQueue(CommandQueue&& other)
{
    if (m_pCommander.lock() != other.m_pCommander.lock())
    {
        assert(false);
        return;
    }

    for (size_t i = 0; i < mcore::enum_size<QueueType>(); ++i)
    {
        m_commandQueues[i].insert(m_commandQueues[i].end(), other.m_commandQueues[i].begin(), other.m_commandQueues[i].end());
    }
    m_waitSemaphores.insert(m_waitSemaphores.end(), 
        std::make_move_iterator(other.m_waitSemaphores.begin()),
        std::make_move_iterator(other.m_waitSemaphores.end()));
    m_tempCommandbuffers.insert(m_tempCommandbuffers.end(),
        std::make_move_iterator(other.m_tempCommandbuffers.begin()),
        std::make_move_iterator(other.m_tempCommandbuffers.end()));
}

mvk::Future mvk::CommandQueue::Flush(
    vk::PipelineStageFlags2 graphicStageMask,
    vk::PipelineStageFlags2 computeStageMask,
    vk::PipelineStageFlags2 transferStageMask)
{
    const auto pCommander = m_pCommander.lock();
    if (!pCommander) return {};

    const auto pCore = VulkanCore::Instance();

    std::array<vk::PipelineStageFlags2, 3> stageMasks = {
        graphicStageMask,
        computeStageMask,
        transferStageMask,
    };
    auto commandIDQueues = std::move(m_commandQueues);
    if (!commandIDQueues[to_sizet(QueueType::COMPUTE)].empty() && pCore->GetGraphicQueue() == pCore->GetComputeQueue())
    {
        commandIDQueues[to_sizet(QueueType::GRAPHIC)].insert(commandIDQueues[to_sizet(QueueType::GRAPHIC)].end(),
            std::make_move_iterator(commandIDQueues[to_sizet(QueueType::COMPUTE)].begin()),
            std::make_move_iterator(commandIDQueues[to_sizet(QueueType::COMPUTE)].end()));
        commandIDQueues[to_sizet(QueueType::COMPUTE)].clear();
    }
    if (!commandIDQueues[to_sizet(QueueType::TRANSFER)].empty() && pCore->GetGraphicQueue() == pCore->GetTransferQueue())
    {
        commandIDQueues[to_sizet(QueueType::GRAPHIC)].insert(commandIDQueues[to_sizet(QueueType::GRAPHIC)].end(),
            std::make_move_iterator(commandIDQueues[to_sizet(QueueType::TRANSFER)].begin()),
            std::make_move_iterator(commandIDQueues[to_sizet(QueueType::TRANSFER)].end()));
        commandIDQueues[to_sizet(QueueType::TRANSFER)].clear();
    }

    std::vector<vk::SemaphoreSubmitInfo> waitSemaphoreInfos;
    waitSemaphoreInfos.reserve(m_waitSemaphores.size());
    std::ranges::transform(m_waitSemaphores, std::back_inserter(waitSemaphoreInfos), [&](const auto& waitSemaphore)
    {
        const auto& [semaphoreID, stageMask] = waitSemaphore;
        return vk::SemaphoreSubmitInfo{
            pCommander->Get(semaphoreID),
            0,
            stageMask
        };
    });

    std::vector<std::tuple<SemaphoreID, vk::PipelineStageFlags2>> signalSemaphores;
    std::vector<FenceID> fences;
    signalSemaphores.reserve(3);
    fences.reserve(3);
    for (size_t i = 0; i < enum_size<QueueType>(); ++i)
    {
        if (commandIDQueues[i].empty()) continue;

        std::vector<vk::CommandBufferSubmitInfo> commandBufferSubmitInfo;
        commandBufferSubmitInfo.reserve(commandIDQueues[i].size());
        std::ranges::transform(commandIDQueues[i], std::back_inserter(commandBufferSubmitInfo), [&](const auto cmdBuffer)
        {
            return vk::CommandBufferSubmitInfo{
                pCommander->Get(cmdBuffer)
            };
        });

        auto signalSemaphoreID = pCommander->CreateSemaphore();
        auto fenceID = pCommander->CreateFence();

        vk::SemaphoreSubmitInfo signalInfo{
            pCommander->Get(signalSemaphoreID),
            0,
            stageMasks[i],
        };
        vk::SubmitInfo2 submitInfo{
            vk::SubmitFlags{},
            waitSemaphoreInfos,
            commandBufferSubmitInfo,
            signalInfo
        };
        pCore->GetQueue(from_sizet<QueueType>(i)).submit2(submitInfo, pCommander->Get(fenceID));

        signalSemaphores.emplace_back(std::move(signalSemaphoreID), stageMasks[i]);
        fences.emplace_back(std::move(fenceID));
    }

    std::transform(
        std::make_move_iterator(m_waitSemaphores.begin()),
        std::make_move_iterator(m_waitSemaphores.end()),
        std::back_inserter(m_tempSemaphores), [&](auto&& waitSemaphore)
    {
        auto [semaphoreID, stageMask] = std::move(waitSemaphore);
        return semaphoreID;
    });

    for (size_t i = 0; i < enum_size<QueueType>(); ++i)
    {
        m_tempCommandbuffers.insert(m_tempCommandbuffers.end(),
            std::make_move_iterator(commandIDQueues[i].begin()),
            std::make_move_iterator(commandIDQueues[i].end()));
    }

    Future future{ m_pCommander, std::move(fences), std::move(m_tempSemaphores), std::move(m_tempCommandbuffers) };
    m_waitSemaphores = std::move(signalSemaphores);
    return future;
}

void mvk::CommandQueue::FlushAsync(
    vk::PipelineStageFlags2 graphicStageMask,
    vk::PipelineStageFlags2 computeStageMask,
    vk::PipelineStageFlags2 transferStageMask)
{
    const auto pCommander = m_pCommander.lock();
    if (!pCommander) return;

    const auto pCore = VulkanCore::Instance();

    std::array<vk::PipelineStageFlags2, 3> stageMasks = {
        graphicStageMask,
        computeStageMask,
        transferStageMask,
    };
    auto commandIDQueues = std::move(m_commandQueues);
    if (!commandIDQueues[to_sizet(QueueType::COMPUTE)].empty() && pCore->GetGraphicQueue() == pCore->GetComputeQueue())
    {
        commandIDQueues[to_sizet(QueueType::GRAPHIC)].insert(commandIDQueues[to_sizet(QueueType::GRAPHIC)].end(),
            std::make_move_iterator(commandIDQueues[to_sizet(QueueType::COMPUTE)].begin()),
            std::make_move_iterator(commandIDQueues[to_sizet(QueueType::COMPUTE)].end()));
        commandIDQueues[to_sizet(QueueType::COMPUTE)].clear();
    }
    if (!commandIDQueues[to_sizet(QueueType::TRANSFER)].empty() && pCore->GetGraphicQueue() == pCore->GetTransferQueue())
    {
        commandIDQueues[to_sizet(QueueType::GRAPHIC)].insert(commandIDQueues[to_sizet(QueueType::GRAPHIC)].end(),
            std::make_move_iterator(commandIDQueues[to_sizet(QueueType::TRANSFER)].begin()),
            std::make_move_iterator(commandIDQueues[to_sizet(QueueType::TRANSFER)].end()));
        commandIDQueues[to_sizet(QueueType::TRANSFER)].clear();
    }

    std::vector<vk::SemaphoreSubmitInfo> waitSemaphoreInfos;
    waitSemaphoreInfos.reserve(m_waitSemaphores.size());
    std::ranges::transform(m_waitSemaphores, std::back_inserter(waitSemaphoreInfos), [&](const auto& waitSemaphore)
    {
        const auto& [semaphoreID, stageMask] = waitSemaphore;
        return vk::SemaphoreSubmitInfo{
            pCommander->Get(semaphoreID),
            0,
            stageMask
        };
    });

    std::vector<std::tuple<SemaphoreID, vk::PipelineStageFlags2>> signalSemaphores;
    signalSemaphores.reserve(3);
    for (size_t i = 0; i < enum_size<QueueType>(); ++i)
    {
        if (commandIDQueues[i].empty()) continue;

        std::vector<vk::CommandBufferSubmitInfo> commandBufferSubmitInfo;
        commandBufferSubmitInfo.reserve(commandIDQueues[i].size());
        std::ranges::transform(commandIDQueues[i], std::back_inserter(commandBufferSubmitInfo), [&](const auto cmdBuffer)
        {
            return vk::CommandBufferSubmitInfo{
                pCommander->Get(cmdBuffer)
            };
        });

        auto signalSemaphoreID = pCommander->CreateSemaphore();

        vk::SemaphoreSubmitInfo signalInfo{
            pCommander->Get(signalSemaphoreID),
            0,
            stageMasks[i],
        };
        vk::SubmitInfo2 submitInfo{
            vk::SubmitFlags{},
            waitSemaphoreInfos,
            commandBufferSubmitInfo,
            signalInfo
        };
        pCore->GetQueue(from_sizet<QueueType>(i)).submit2(submitInfo);

        signalSemaphores.emplace_back(std::move(signalSemaphoreID), stageMasks[i]);
    }

    std::transform(
        std::make_move_iterator(m_waitSemaphores.begin()),
        std::make_move_iterator(m_waitSemaphores.end()),
        std::back_inserter(m_tempSemaphores), [&](auto&& waitSemaphore)
    {
        auto [semaphoreID, stageMask] = std::move(waitSemaphore);
        return semaphoreID;
    });
    
    for (size_t i = 0; i < enum_size<QueueType>(); ++i)
    {
        m_tempCommandbuffers.insert(m_tempCommandbuffers.end(),
            std::make_move_iterator(commandIDQueues[i].begin()),
            std::make_move_iterator(commandIDQueues[i].end()));
    }
    m_waitSemaphores = std::move(signalSemaphores);
}

mvk::Future::Future(const std::weak_ptr<Commander>& pCommander,
    std::vector<FenceID>&& fences,
    std::vector<SemaphoreID>&& waitingSemaphores,
    std::vector<CommandID>&& waitingCommandQueues)
    : m_pCommander{ pCommander }
    , m_fences{ fences }
    , m_waitingSemaphores{ std::move(waitingSemaphores) }
    , m_waitingCommandQueues{ std::move(waitingCommandQueues) }
{}

void mvk::Future::Reset()
{
    m_pCommander.reset();
}

void mvk::Future::Wait(uint64_t timeout)
{
    const auto pCommander = m_pCommander.lock();
    if (!pCommander) return;

    std::vector<vk::Fence> fences;
    fences.reserve(m_fences.size());
    std::ranges::transform(m_fences, std::back_inserter(fences), [&](const auto& fenceID)
    {
        return pCommander->Get(fenceID);
    });

    const auto pCore = VulkanCore::Instance();
    const auto result = pCore->GetDevice().waitForFences(fences, true, timeout);

    m_fences.clear();
    m_waitingSemaphores.clear();
    m_waitingCommandQueues.clear();
    m_pCommander.reset();
}

mvk::Commander::Commander(const std::weak_ptr<mvk::RenderContext>& pRenderContext)
    : m_pRenderContext{ pRenderContext }
{
    for (auto& commandbufferArrays : m_commandbuffers)
    {
        for (auto& commandbufferArray : commandbufferArrays)
        {
            commandbufferArray.reserve(16);
        }
    }
}

mvk::Commander::~Commander()
{
    if (const auto pRenderContext = m_pRenderContext.lock(); pRenderContext)
    {
        Destroy();
    }
}

void mvk::Commander::Initialize()
{
    const auto pCore = VulkanCore::Instance();

    for (const auto& [commandType, flags] : TYPE_TO_FLAGBITS)
    {
        m_pools[to_sizet(QueueType::GRAPHIC)][to_sizet(commandType)]
            = CreateCommandPool(pCore->GetQueueFamilyIndices().Get(QueueType::GRAPHIC), flags);

        if (pCore->GetQueueFamilyIndices().GetGraphics() != pCore->GetQueueFamilyIndices().GetCompute())
        {
            m_pools[to_sizet(QueueType::COMPUTE)][to_sizet(commandType)]
                = CreateCommandPool(pCore->GetQueueFamilyIndices().Get(QueueType::COMPUTE), flags);
        }
        else
        {
            m_pools[to_sizet(QueueType::COMPUTE)][to_sizet(commandType)]
                = m_pools[to_sizet(QueueType::GRAPHIC)][to_sizet(commandType)];
        }
        if (pCore->GetQueueFamilyIndices().GetGraphics() != pCore->GetQueueFamilyIndices().GetTransfer())
        {
            m_pools[to_sizet(QueueType::TRANSFER)][to_sizet(commandType)]
                = CreateCommandPool(pCore->GetQueueFamilyIndices().Get(QueueType::TRANSFER), flags);
        }
        else
        {
            m_pools[to_sizet(QueueType::TRANSFER)][to_sizet(commandType)]
                = m_pools[to_sizet(QueueType::GRAPHIC)][to_sizet(commandType)];
        }
    }
}

void mvk::Commander::Destroy()
{
    const auto pCore = VulkanCore::Instance();
    for (auto& commandbufferArrays : m_commandbuffers)
    {
        for (auto& commandbufferArray : commandbufferArrays)
        {
            commandbufferArray.clear();
        }
    }
    for (auto& poolArray : m_pools)
    {
        for (auto& pool : poolArray)
        {
            pCore->GetDevice().destroyCommandPool(pool);
        }
    }
    for (auto& semaphore : m_semaphores)
    {
        pCore->GetDevice().destroySemaphore(semaphore);
    }
    m_semaphores.clear();
    for (auto& fence : m_fences)
    {
        pCore->GetDevice().destroyFence(fence);
    }
    m_fences.clear();
}

mvk::CommandID mvk::Commander::CreateCommandBuffer(QueueType queueType, CommandType commandType)
{
    const auto pCore = VulkanCore::Instance();

    if (!m_idleCommandBuffers.empty())
    {
        CommandID commandID{ shared_from_this(), m_idleCommandBuffers.back() };
        m_idleCommandBuffers.pop_back();
        Get(commandID).reset();
        return commandID;
    }

    const auto index = m_commandbuffers[to_sizet(queueType)][to_sizet(commandType)].size();
    vk::CommandBufferAllocateInfo allocInfo{
        m_pools[to_sizet(queueType)][to_sizet(commandType)],
        vk::CommandBufferLevel::ePrimary,
        commandType == CommandType::DYNAMIC ? GetRenderContext()->GetMaxFramesinFlight() : 1
    };
    m_commandbuffers[to_sizet(queueType)][to_sizet(commandType)].emplace_back(pCore->GetDevice().allocateCommandBuffers(allocInfo));
    m_commandIDCounts[to_sizet(queueType)][to_sizet(commandType)].emplace_back(0);
    return { shared_from_this(), { queueType, commandType, index } };
}

mvk::QueueType mvk::Commander::GetQueueType(const CommandID& commandID) const
{
    return commandID.m_data.queueType;
}

mvk::CommandType mvk::Commander::GetCommandType(const CommandID& commandID) const
{
    return commandID.m_data.commandType;
}

vk::CommandBuffer mvk::Commander::Get(const CommandID& commandID) const
{
    return m_commandbuffers[to_sizet(commandID.m_data.queueType)][to_sizet(commandID.m_data.commandType)][commandID.m_data.index]
        [commandID.m_data.commandType == CommandType::DYNAMIC ? GetRenderContext()->GetInFlightIndex() : 0];
}

mvk::SemaphoreID mvk::Commander::CreateSemaphore()
{
    const auto pCore = VulkanCore::Instance();

    if (!m_idleSemaphores.empty())
    {
        SemaphoreID semaphoreID{ shared_from_this(), m_idleSemaphores.back() };
        m_idleSemaphores.pop_back();
        return semaphoreID;
    }

    const auto index = m_semaphores.size();
    vk::SemaphoreCreateInfo createInfo{};
    m_semaphores.emplace_back(pCore->GetDevice().createSemaphore(createInfo));
    m_semaphoreIDCounts.emplace_back(0);
    return { shared_from_this(), index };
}

vk::Semaphore mvk::Commander::Get(const SemaphoreID& semaphoreID) const
{
    return m_semaphores[semaphoreID.m_index];
}

mvk::FenceID mvk::Commander::CreateFence()
{
    const auto pCore = VulkanCore::Instance();

    if (!m_idleFences.empty())
    {
        FenceID fenceID{ shared_from_this(), m_idleFences.back() };
        m_idleFences.pop_back();
        pCore->GetDevice().resetFences(Get(fenceID));
        return fenceID;
    }

    const auto index = m_fences.size();
    vk::FenceCreateInfo fenceCreateInfo{};
    m_fences.emplace_back(pCore->GetDevice().createFence(fenceCreateInfo));
    m_fenceIDCounts.emplace_back(0);
    return { shared_from_this(), index };
}

vk::Fence mvk::Commander::Get(const FenceID& fenceID) const
{
    return m_fences[fenceID.m_index];
}

vk::CommandPool mvk::Commander::CreateCommandPool(uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags flags)
{
    vk::CommandPoolCreateInfo poolInfo{
        flags,
        queueFamilyIndex,
    };
    return VulkanCore::Instance()->GetDevice().createCommandPool(poolInfo);
}
