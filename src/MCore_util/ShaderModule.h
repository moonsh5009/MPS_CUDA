#pragma once

#include "VulkanDef.h"

#include "HeaderPre.h"

namespace mvk
{
	class __MY_EXT_CLASS__ ShaderModule
	{
	public:
		ShaderModule(const std::string& path);
		~ShaderModule();
		ShaderModule(const ShaderModule&) = delete;
		ShaderModule(ShaderModule&& other) noexcept;
		ShaderModule& operator=(const ShaderModule&) = delete;
		ShaderModule& operator=(ShaderModule&& other) noexcept;

		void SetEntryPoint(const std::string& entryPoint);
		void SetStage(vk::ShaderStageFlagBits stage);
		void SetConstantEntries(std::vector<vk::SpecializationMapEntry>&& constantEntries);
		void SetConstantData(std::vector<char>&& constantData);
		vk::PipelineShaderStageCreateInfo Get() const;

		constexpr const vk::ShaderModule& GetModule() const { return m_module; }

	private:
		vk::ShaderModule m_module = {};
		std::string m_entryPoint = {};
		std::vector<vk::SpecializationMapEntry> m_constantEntries = {};
		std::vector<char> m_constantData = {};

		vk::SpecializationInfo m_specializationInfo = {};
		vk::PipelineShaderStageCreateInfo m_shaderStageCreateInfo = {};
	};
}

#include "HeaderPost.h"