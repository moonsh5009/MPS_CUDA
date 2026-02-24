#include "stdafx.h"
#include "WindowEventInfo.h"

std::tuple<int, int> mcore::GetMousePosition(LPARAM lParam)
{
    return { ((int)(short)LOWORD(lParam)), ((int)(short)HIWORD(lParam)) };
}

mcore::MouseModifiers mcore::GetMouseModifiers(WPARAM wParam)
{
    MouseModifiers mods = MouseModifierBits::None;

    if (wParam & MK_SHIFT)
        mods = mods | MouseModifierBits::Shift;
    if (wParam & MK_CONTROL)
        mods = mods | MouseModifierBits::Ctrl;
    // Alt는 GetKeyState로 확인
    if (GetKeyState(VK_MENU) & 0x8000)
        mods = mods | MouseModifierBits::Alt;
    if (GetKeyState(VK_LWIN) & 0x8000 || GetKeyState(VK_RWIN) & 0x8000)
        mods = mods | MouseModifierBits::Meta;

    return mods;
}

mcore::MouseEventInfo mcore::CreateMouseEventInfo(WPARAM wParam, LPARAM lParam, MouseButton button)
{
    MouseEventInfo info;
    std::tie(info.x, info.y) = GetMousePosition(lParam);
    info.button = button;
    info.modifiers = GetMouseModifiers(wParam);
    return info;
}

mcore::MouseEventInfo mcore::CreateMouseMoveEventInfo(WPARAM wParam, LPARAM lParam)
{
    MouseEventInfo info;
    std::tie(info.x, info.y) = GetMousePosition(lParam);

    // 눌린 버튼들 체크
    info.button = MouseButtonBits::None;
    if (wParam & MK_LBUTTON)
        info.button = info.button | MouseButtonBits::Left;
    if (wParam & MK_RBUTTON)
        info.button = info.button | MouseButtonBits::Right;
    if (wParam & MK_MBUTTON)
        info.button = info.button | MouseButtonBits::Middle;
    if (wParam & MK_XBUTTON1)
        info.button = info.button | MouseButtonBits::X1;
    if (wParam & MK_XBUTTON2)
        info.button = info.button | MouseButtonBits::X2;

    info.modifiers = GetMouseModifiers(wParam);
    return info;
}

mcore::MouseEventInfo mcore::CreateMouseWheelEventInfo(HWND hWnd, WPARAM wParam, LPARAM lParam)
{
    POINT pt;
    std::tie(pt.x, pt.y) = GetMousePosition(lParam);
    ScreenToClient(hWnd, &pt);

    MouseEventInfo info;
    info.x = pt.x;
    info.y = pt.y;

    // HIWORD(wParam)에 휠 델타값이 들어있음
    // WHEEL_DELTA (120) 단위
    info.wheelDelta = static_cast<int>(static_cast<short>(HIWORD(wParam)));

    info.modifiers = GetMouseModifiers(wParam);
    return info;
}

mcore::KeyEventInfo mcore::CreateKeyEventInfo(WPARAM wParam, LPARAM lParam)
{
    KeyEventInfo info;

    info.keyCode = static_cast<unsigned int>(wParam);
    info.repeatCount = static_cast<unsigned int>(lParam & 0xFFFF);

    unsigned scanCode = static_cast<unsigned int>((lParam >> 16) & 0xFF);
    bool isExtended = (lParam & (1 << 24)) != 0;
    bool previousKeyState = (lParam & (1 << 30)) != 0;

    info.flags = KeyFlagBits::None;
    if (previousKeyState || info.repeatCount > 1)
        info.flags = info.flags | KeyFlagBits::Repeat;
    if (isExtended)
        info.flags = info.flags | KeyFlagBits::Extended;
    if (!isExtended && wParam >= VK_NUMPAD0 && wParam <= VK_DIVIDE)
        info.flags = info.flags | KeyFlagBits::NumPad;
    if (wParam == VK_PROCESSKEY)
        info.flags = info.flags | KeyFlagBits::IsComposing;
    switch (wParam)
    {
    case VK_SHIFT:
        if (scanCode == 0x2A)
            info.flags = info.flags | KeyFlagBits::LeftMod;
        else if (scanCode == 0x36)
            info.flags = info.flags | KeyFlagBits::RightMod;
        break;

    case VK_CONTROL:
        if (isExtended)
            info.flags = info.flags | KeyFlagBits::RightMod;
        else
            info.flags = info.flags | KeyFlagBits::LeftMod;
        break;

    case VK_MENU:  // Alt
        if (isExtended)
            info.flags = info.flags | KeyFlagBits::RightMod;
        else
            info.flags = info.flags | KeyFlagBits::LeftMod;
        break;
    }

    info.modifiers = KeyModifierBits::None;
    if (GetKeyState(VK_SHIFT) & 0x8000)
        info.modifiers = info.modifiers | KeyModifierBits::Shift;
    if (GetKeyState(VK_CONTROL) & 0x8000)
        info.modifiers = info.modifiers | KeyModifierBits::Ctrl;
    if (GetKeyState(VK_MENU) & 0x8000)
        info.modifiers = info.modifiers | KeyModifierBits::Alt;
    if (GetKeyState(VK_CAPITAL) & 0x0001)
        info.modifiers = info.modifiers | KeyModifierBits::CapsLock;
    if (GetKeyState(VK_NUMLOCK) & 0x0001)
        info.modifiers = info.modifiers | KeyModifierBits::NumLock;
    if ((GetKeyState(VK_LWIN) & 0x8000) || (GetKeyState(VK_RWIN) & 0x8000))
        info.modifiers = info.modifiers | KeyModifierBits::Meta;

    return info;
}