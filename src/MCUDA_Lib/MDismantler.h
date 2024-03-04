#pragma once

template<typename CallbackFunc>
class MDismantler final
{
public:
	MDismantler() noexcept {}
	MDismantler(CallbackFunc callback) noexcept : m_callback{ callback } {}
	~MDismantler() { m_callback(); }

private:
	CallbackFunc m_callback;
};