#pragma once

#include "VulkanDef.h"

#include "../MCore_util/EnumExt.h"

#include "HeaderPre.h"

#define USE_VULKAN_SUBMIT_2     1

namespace mvk
{
    class RenderContext;
    class Commander;
    class Future;

    struct CommandIDData
    {
        QueueType queueType;
        CommandType commandType;
        size_t index;
    };

    class __MY_EXT_CLASS__ CommandID
    {
    public:
        CommandID() = default;
        CommandID(const std::shared_ptr<Commander>& pCommander, CommandIDData data);
        ~CommandID();
        CommandID(const CommandID& other) { *this = other; }
        CommandID(CommandID&&) = default;
        CommandID& operator=(const CommandID& other);
        CommandID& operator=(CommandID&&) = default;

        void Destroy();

        operator bool() const
        {
            return !m_pCommander.expired();
        }

    private:
        friend class Commander;
        std::weak_ptr<Commander> m_pCommander;
        CommandIDData m_data;
    };

    class __MY_EXT_CLASS__ SemaphoreID
    {
    public:
        SemaphoreID() = default;
        SemaphoreID(const std::shared_ptr<Commander>& pCommander, size_t index);
        ~SemaphoreID();
        SemaphoreID(const SemaphoreID& other) { *this = other; }
        SemaphoreID(SemaphoreID&&) = default;
        SemaphoreID& operator=(const SemaphoreID& other);
        SemaphoreID& operator=(SemaphoreID&&) = default;

        void Destroy();

        operator bool() const
        {
            return !m_pCommander.expired();
        }

    private:
        friend class Commander;
        std::weak_ptr<Commander> m_pCommander;
        size_t m_index;
    };

    class __MY_EXT_CLASS__ FenceID
    {
    public:
        FenceID() = default;
        FenceID(const std::shared_ptr<Commander>& pCommander, size_t index);
        ~FenceID();
        FenceID(const FenceID& other) { *this = other; }
        FenceID(FenceID&&) = default;
        FenceID& operator=(const FenceID& other);
        FenceID& operator=(FenceID&&) = default;

        void Destroy();

        operator bool() const
        {
            return !m_pCommander.expired();
        }

    private:
        friend class Commander;
        std::weak_ptr<Commander> m_pCommander;
        size_t m_index;
    };

    class __MY_EXT_CLASS__ CommandQueue
    {
    public:
        CommandQueue() = delete;
        CommandQueue(const std::weak_ptr<Commander>& pCommander);
        ~CommandQueue() = default;
        CommandQueue(const CommandQueue&) = delete;
        CommandQueue(CommandQueue&& other) = default;
        CommandQueue& operator=(const CommandQueue&) = delete;
        CommandQueue& operator=(CommandQueue&& other) = default;

        void Reset();

        void AddSemaphore(const SemaphoreID& semaphoreID, vk::PipelineStageFlags2 stageMask);
        void AddCommand(const CommandID& commandID);
        void AddCommand(CommandID&& commandID);
        void AddCommandQueue(CommandQueue&& other);
        Future Flush(
            vk::PipelineStageFlags2 graphicStageMask,
            vk::PipelineStageFlags2 computeStageMask = {},
            vk::PipelineStageFlags2 transferStageMask = {});
        void FlushAsync(
            vk::PipelineStageFlags2 graphicStageMask,
            vk::PipelineStageFlags2 computeStageMask = {},
            vk::PipelineStageFlags2 transferStageMask = {});

        std::vector<std::tuple<SemaphoreID, vk::PipelineStageFlags2>> GetWaitSemaphores()
        {
            return std::move(m_waitSemaphores);
        }

    private:
        friend class Commander;
        friend class Future;
        std::weak_ptr<Commander> m_pCommander;

        std::array<std::vector<CommandID>, mcore::enum_size<QueueType>()> m_commandQueues;
        std::vector<std::tuple<SemaphoreID, vk::PipelineStageFlags2>> m_waitSemaphores;

        std::vector<CommandID> m_tempCommandbuffers;
        std::vector<SemaphoreID> m_tempSemaphores;
    };

    class __MY_EXT_CLASS__ Future
    {
    public:
        Future() = default;
        Future(const std::weak_ptr<Commander>& pCommander,
            std::vector<FenceID>&& fences,
            std::vector<SemaphoreID>&& waitingSemaphores,
            std::vector<CommandID>&& waitingCommandQueues);
        ~Future() = default;
        Future(const Future&) = delete;
        Future(Future&&) = default;
        Future& operator=(const Future&) = delete;
        Future& operator=(Future&&) = default;

        void Reset();

        void Wait(uint64_t timeout = UINT64_MAX);

    private:
        std::weak_ptr<Commander> m_pCommander;

        std::vector<FenceID> m_fences;

        std::vector<SemaphoreID> m_waitingSemaphores;
        std::vector<CommandID> m_waitingCommandQueues;
    };

    class __MY_EXT_CLASS__ Commander : public std::enable_shared_from_this<Commander>
    {
    public:
        Commander() = delete;
        Commander(const std::weak_ptr<mvk::RenderContext>& pRenderContext);
        ~Commander();
        Commander(const Commander&) = delete;
        Commander(Commander&&) = default;
        Commander& operator=(const Commander&) = delete;
        Commander& operator=(Commander&&) = default;

        void Initialize();
        void Destroy();

        CommandID CreateCommandBuffer(QueueType queueType, CommandType commandType);
        QueueType GetQueueType(const CommandID& commandID) const;
        CommandType GetCommandType(const CommandID& commandID) const;
        vk::CommandBuffer Get(const CommandID& commandID) const;

        SemaphoreID CreateSemaphore();
        vk::Semaphore Get(const SemaphoreID& semaphoreID) const;

        FenceID CreateFence();
        vk::Fence Get(const FenceID& fenceID) const;

        std::shared_ptr<mvk::RenderContext> GetRenderContext() const
        {
            return m_pRenderContext.lock();
        }

    private:
        friend class CommandID;
        friend class SemaphoreID;
        friend class FenceID;
        vk::CommandPool CreateCommandPool(uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags flags);

        std::weak_ptr<mvk::RenderContext> m_pRenderContext;

        std::array<std::array<vk::CommandPool,
            mcore::enum_size<CommandType>()>, mcore::enum_size<QueueType>()> m_pools;
        std::array<std::array<std::vector<std::vector<vk::CommandBuffer>>,
            mcore::enum_size<CommandType>()>, mcore::enum_size<QueueType>()> m_commandbuffers;
        std::array<std::array<std::vector<uint32_t>,
            mcore::enum_size<CommandType>()>, mcore::enum_size<QueueType>()> m_commandIDCounts;

        std::vector<vk::Semaphore> m_semaphores;
        std::vector<uint32_t> m_semaphoreIDCounts;

        std::vector<vk::Fence> m_fences;
        std::vector<uint32_t> m_fenceIDCounts;

        std::vector<CommandIDData> m_idleCommandBuffers;
        std::vector<size_t> m_idleSemaphores;
        std::vector<size_t> m_idleFences;
    };
}

#include "HeaderPost.h"