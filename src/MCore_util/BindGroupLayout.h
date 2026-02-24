#pragma once

#include "VKDeviceMemory.h"

#include "HeaderPre.h"

namespace mvk
{
    template<typename T>
    concept BufferLike = requires(T t)
    {
        { t.GetBuffer() } -> std::convertible_to<vk::Buffer>;
        { t.GetByteLength() } -> std::convertible_to<vk::DeviceSize>;
    };

    template<typename T>
    concept ImageLike = requires(T t)
    {
        { t.GetView() } -> std::convertible_to<vk::ImageView>;
    };

    class BindGroupLayoutImpl;
    class __MY_EXT_CLASS__ BindGroupLayoutBatchBinder
    {
    public:
        explicit BindGroupLayoutBatchBinder(const BindGroupLayoutImpl* layout);
        ~BindGroupLayoutBatchBinder() = default;
        BindGroupLayoutBatchBinder(const BindGroupLayoutBatchBinder&) = delete;
        BindGroupLayoutBatchBinder(BindGroupLayoutBatchBinder&&) = default;
        BindGroupLayoutBatchBinder& operator=(const BindGroupLayoutBatchBinder&) = delete;
        BindGroupLayoutBatchBinder& operator=(BindGroupLayoutBatchBinder&&) = default;

        template<BufferLike T>
        BindGroupLayoutBatchBinder&& SetUniformBuffer(uint32_t binding,
            const T& buffer, uint32_t instance = 0)&&
        {
            return std::move(*this).SetBuffer(binding, buffer.GetBuffer(), buffer.GetByteLength(),
                vk::DescriptorType::eUniformBuffer, instance);
        }

        template<BufferLike T>
        BindGroupLayoutBatchBinder&& SetStorageBuffer(uint32_t binding,
            const T& buffer, uint32_t instance = 0)&&
        {
            return std::move(*this).SetBuffer(binding, buffer.GetBuffer(), buffer.GetByteLength(),
                vk::DescriptorType::eStorageBuffer, instance);
        }

        template<BufferLike T>
        BindGroupLayoutBatchBinder&& SetDynamicUniformBuffer(uint32_t binding,
            const T& buffer, uint32_t instance = 0)&&
        {
            return std::move(*this).SetBuffer(binding, buffer.GetBuffer(), buffer.GetByteLength(),
                vk::DescriptorType::eUniformBufferDynamic, instance);
        }

        template<ImageLike T>
        BindGroupLayoutBatchBinder&& SetImage(uint32_t binding, vk::Sampler sampler,
            const T& image, vk::ImageLayout layout, uint32_t instance = 0)&&
        {
            return std::move(*this).SetImage(binding, sampler, image.GetImageView(), layout, instance);
        }

        template<ImageLike T>
        BindGroupLayoutBatchBinder&& SetStorageImage(uint32_t binding,
            const T& image, vk::ImageLayout layout, uint32_t instance = 0)&&
        {
            return std::move(*this).SetStorageImage(binding, image.GetImageView(), layout, instance);
        }

        BindGroupLayoutBatchBinder&& Reserve(size_t count)&&;
        BindGroupLayoutBatchBinder&& SetNone(uint32_t binding, vk::DescriptorType type, uint32_t instance = 0)&&;
        BindGroupLayoutBatchBinder&& SetBuffer(uint32_t binding, vk::Buffer buffer,
            vk::DeviceSize size, vk::DescriptorType type, uint32_t instance = 0)&&;
        BindGroupLayoutBatchBinder&& SetImage(uint32_t binding, vk::Sampler sampler,
            vk::ImageView imageView, vk::ImageLayout layout, uint32_t instance = 0)&&;
        BindGroupLayoutBatchBinder&& SetStorageImage(uint32_t binding, vk::ImageView imageView,
            vk::ImageLayout layout, uint32_t instance = 0)&&;
        void Commit()&&;

    private:
        const BindGroupLayoutImpl* m_layout;
        std::vector<std::tuple<vk::DescriptorBufferInfo, vk::DescriptorType, uint32_t, uint32_t>> m_bufferInfos;
        std::vector<std::tuple<vk::DescriptorImageInfo, vk::DescriptorType, uint32_t, uint32_t>> m_imageInfos;
    };

    class __MY_EXT_CLASS__ BindGroupLayoutImpl
    {
    public:
        BindGroupLayoutImpl() = delete;
        BindGroupLayoutImpl(
            vk::DescriptorPool pool,
            vk::DescriptorSetLayout layout,
            std::vector<vk::DescriptorSet>&& sets,
            std::vector<vk::DescriptorSetLayoutBinding>&& bindings,
            std::vector<std::string>&& bindingLabels);
        ~BindGroupLayoutImpl();
        BindGroupLayoutImpl(const BindGroupLayoutImpl&) = default;
        BindGroupLayoutImpl(BindGroupLayoutImpl&&) = default;
        BindGroupLayoutImpl& operator=(const BindGroupLayoutImpl&) = default;
        BindGroupLayoutImpl& operator=(BindGroupLayoutImpl&&) = default;

        void Destroy();

        BindGroupLayoutBatchBinder Binder() const
        {
            return BindGroupLayoutBatchBinder(this);
        }

        template<BufferLike T>
        void UpdateUniformBuffer(uint32_t binding, const T& buffer, uint32_t instance = 0) const
        {
            Binder()
                .SetUniformBuffer(binding, buffer, instance)
                .Commit();
        }

        template<BufferLike T>
        void UpdateStorageBuffer(uint32_t binding, const T& buffer, uint32_t instance = 0) const
        {
            Binder()
                .SetStorageBuffer(binding, buffer, instance)
                .Commit();
        }

        template<ImageLike T>
        void UpdateImage(uint32_t binding, vk::Sampler sampler,
            const T& image, vk::ImageLayout layout, uint32_t instance = 0) const
        {
            Binder()
                .SetImage(binding, sampler, image, layout, instance)
                .Commit();
        }

        template<ImageLike T>
        void UpdateStorageImage(uint32_t binding, const T& image,
            vk::ImageLayout layout, uint32_t instance = 0) const
        {
            Binder()
                .SetStorageImage(binding, image, layout, instance)
                .Commit();
        }

        std::string GetDescription() const;

        bool IsValid() const { return m_layout; }
        uint32_t GetInstanceCount() const { return static_cast<uint32_t>(m_sets.size()); }

        operator vk::DescriptorSetLayout() const { return m_layout; }
        vk::DescriptorSetLayout Get() const { return m_layout; }
        vk::DescriptorSet GetSet(uint32_t instanceIndex = 0) const { return m_sets[instanceIndex]; }
        vk::DescriptorPool GetPool() const { return m_pool; }

    private:
        friend class BindGroupLayoutBatchBinder;

        vk::DescriptorPool m_pool;
        vk::DescriptorSetLayout m_layout;
        std::vector<vk::DescriptorSet> m_sets;

        std::vector<vk::DescriptorSetLayoutBinding> m_bindings;
        std::vector<std::string> m_bindingLabels;
    };
    using BindGroupLayout = std::shared_ptr<BindGroupLayoutImpl>;

    class __MY_EXT_CLASS__ BindGroupLayoutBuilder final
    {
    public:
        BindGroupLayoutBuilder() = default;
        ~BindGroupLayoutBuilder() = default;
        BindGroupLayoutBuilder(const BindGroupLayoutBuilder&) = delete;
        BindGroupLayoutBuilder(BindGroupLayoutBuilder&&) = default;
        BindGroupLayoutBuilder& operator=(const BindGroupLayoutBuilder&) = delete;
        BindGroupLayoutBuilder& operator=(BindGroupLayoutBuilder&&) = default;

        BindGroupLayoutBuilder&& Reserve(uint32_t size)&&;

        BindGroupLayoutBuilder&& AddBinding(uint32_t binding, vk::DescriptorType type,
            vk::ShaderStageFlags stageFlags, uint32_t count = 1, const std::string& label = "")&&;

        BindGroupLayoutBuilder&& AddUniformBuffer(uint32_t binding, vk::ShaderStageFlags stageFlags,
            const std::string& label = "")&&;
        BindGroupLayoutBuilder&& AddStorageBuffer(uint32_t binding, vk::ShaderStageFlags stageFlags,
            const std::string& label = "")&&;
        BindGroupLayoutBuilder&& AddCombinedImageSampler(uint32_t binding, vk::ShaderStageFlags stageFlags,
            uint32_t count = 1, const std::string& label = "")&&;
        BindGroupLayoutBuilder&& AddStorageImage(uint32_t binding, vk::ShaderStageFlags stageFlags,
            const std::string& label = "")&&;

        BindGroupLayout Build(uint32_t numInstances = 1)&&;

        std::string GetDescription() const;

    private:
        std::vector<vk::DescriptorSetLayoutBinding> m_bindings;
        std::vector<std::string> m_bindingLabels;
        vk::DescriptorPoolCreateFlags m_poolFlags = {};

        void ValidateCurrentSet() const;
        std::vector<vk::DescriptorPoolSize> CalculatePoolSizes(uint32_t numInstances) const;
    };
}

#include "HeaderPost.h"