#pragma once

#include "EnumExt.h"
#include <Windows.h>

namespace mcore
{
    enum class MouseButtonBits : unsigned int
    {
        None = 0x00,
        Left = 0x01,
        Right = 0x02,
        Middle = 0x04,
        X1 = 0x08,  // 뒤로가기
        X2 = 0x10   // 앞으로가기
    };
    ENUM_BIT_EXTENSION(MouseButtonBits, MouseButton)
}

namespace mcore
{
    enum class MouseModifierBits : unsigned int
    {
        None = 0x00,
        Shift = 0x01,
        Ctrl = 0x02,
        Alt = 0x04,
        Meta = 0x08
    };
    ENUM_BIT_EXTENSION(MouseModifierBits, MouseModifiers)
}

namespace mcore
{
    enum class KeyFlagBits : unsigned int
    {
        None = 0x0000,
        Repeat = 0x0001,
        Extended = 0x0002,
        NumPad = 0x0004,
        LeftMod = 0x0008,
        RightMod = 0x0010,
        IsComposing = 0x0020
    };
    ENUM_BIT_EXTENSION(KeyFlagBits, KeyFlags)
}

namespace mcore
{
    enum class KeyModifierBits : unsigned int
    {
        None = 0x00,
        Shift = 0x01,
        Ctrl = 0x02,
        Alt = 0x04,
        CapsLock = 0x08,
        NumLock = 0x10,
        Meta = 0x20   // Windows키 / Command키
    };
    ENUM_BIT_EXTENSION(KeyModifierBits, KeyModifiers)
}

#include "HeaderPre.h"

namespace mcore
{
    struct MouseEventInfo
    {
        int x = 0;              // 마우스 X 좌표
        int y = 0;              // 마우스 Y 좌표
        int wheelDelta = 0;     // 휠 스크롤 양 (wheel 이벤트에서 사용)
        MouseButton button = MouseButtonBits::None; // 눌린 버튼
        MouseModifiers modifiers = MouseModifierBits::None; // Shift, Ctrl 등
    };
    struct KeyEventInfo
    {
        unsigned keyCode = 0;
        unsigned repeatCount = 0;
        KeyFlags flags = KeyFlagBits::None;
        KeyModifiers modifiers = KeyModifierBits::None;
    };

    __MY_EXT_CLASS__ std::tuple<int, int> GetMousePosition(LPARAM lParam);
    __MY_EXT_CLASS__ MouseModifiers GetMouseModifiers(WPARAM wParam);
    __MY_EXT_CLASS__ MouseEventInfo CreateMouseEventInfo(WPARAM wParam, LPARAM lParam, MouseButton button);
    __MY_EXT_CLASS__ MouseEventInfo CreateMouseMoveEventInfo(WPARAM wParam, LPARAM lParam);
    __MY_EXT_CLASS__ MouseEventInfo CreateMouseWheelEventInfo(HWND hWnd, WPARAM wParam, LPARAM lParam);
    __MY_EXT_CLASS__ KeyEventInfo CreateKeyEventInfo(WPARAM wParam, LPARAM lParam);
}

#include "HeaderPost.h"