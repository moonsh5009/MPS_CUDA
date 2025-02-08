#include "pch.h"
#include "framework.h"
#include "App.h"
#include "AppDlg.h"
#include "afxdialogex.h"

#include "GL/GLEW/glew.h"
#include "GL/GLEW/wglew.h"
#include "GL/GLFW/glfw3.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CAppDlg::CAppDlg(CWnd* pParent /*=nullptr*/) :
	CDialogEx(IDD_APP_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CAppDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);

	DDX_Control(pDX, IDC_PICTURE, m_picture);
}

BEGIN_MESSAGE_MAP(CAppDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_DESTROY()
	ON_WM_TIMER()
	ON_WM_LBUTTONDOWN()
	ON_WM_RBUTTONDOWN()
	ON_WM_MBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_RBUTTONUP()
	ON_WM_MBUTTONUP()
	ON_WM_MOUSEMOVE()
	ON_WM_MOUSEWHEEL()
END_MESSAGE_MAP()


// CAppDlg 메시지 처리기

BOOL CAppDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	m_pMPSCore = std::make_shared<mps::System>();
	m_pMPSCore->SetDevice(0);

	SetIcon(m_hIcon, TRUE);
	SetIcon(m_hIcon, FALSE);

	if (!GetRenderingContext())
	{
		AfxMessageBox(CString("OpenGL 초기화중 에러가 발생하여 프로그램을 실행할 수 없습니다."));
		return -1;
	}

	m_pMPSCore->Initalize();

	SetTimer(1000, 30, NULL);

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CAppDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CAppDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CAppDlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	if (FALSE == ::wglDeleteContext(m_hRC))
	{
		AfxMessageBox(CString("wglDeleteContext failed"));
	}
}

void CAppDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	m_pMPSCore->Update();
	m_pMPSCore->Draw();

	SwapBuffers(m_pDC->GetSafeHdc());

	CDialogEx::OnTimer(nIDEvent);
}

BOOL CAppDlg::GetRenderingContext()
{
	//픽처 컨트롤에만 그리도록 DC 생성
	//참고 https://goo.gl/CK36zE

	CWnd* pImage = GetDlgItem(IDC_PICTURE);
	CRect rc;
	pImage->GetWindowRect(rc);
	m_pDC = pImage->GetDC();

	if (NULL == m_pDC)
	{
		AfxMessageBox(CString("Unable to get a DC"));
		return FALSE;
	}

	if (!GetOldStyleRenderingContext())
	{
		return TRUE;
	}

	glewExperimental = GL_TRUE;
	if (GLEW_OK != glewInit())
	{
		AfxMessageBox(CString("GLEW could not be initialized!"));
		return FALSE;
	}

	GLint attribs[] =
	{
		WGL_CONTEXT_MAJOR_VERSION_ARB, 2,
		WGL_CONTEXT_MINOR_VERSION_ARB, 0,
		WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
		0
	};

	HGLRC CompHRC = wglCreateContextAttribsARB(m_pDC->GetSafeHdc(), 0, attribs);
	if (CompHRC && wglMakeCurrent(m_pDC->GetSafeHdc(), CompHRC))
		m_hRC = CompHRC;

	return TRUE;
}

BOOL CAppDlg::GetOldStyleRenderingContext()
{
	//A generic pixel format descriptor. This will be replaced with a more
	//specific and modern one later, so don't worry about it too much.
	static PIXELFORMATDESCRIPTOR pfd =
	{
		sizeof(PIXELFORMATDESCRIPTOR),
		1,
		PFD_DRAW_TO_WINDOW |            // support window
		PFD_SUPPORT_OPENGL |            // support OpenGL
		PFD_DOUBLEBUFFER,               // double buffered
		PFD_TYPE_RGBA,                  // RGBA type
		32,                             // 32-bit color depth
		0, 0, 0, 0, 0, 0,               // color bits ignored
		0,								// no alpha buffer
		0,                              // shift bit ignored
		0,                              // no accumulation buffer
		0, 0, 0, 0,                     // accum bits ignored
		24,								// 24-bit z-buffer
		0,								// no stencil buffer
		0,                              // no auxiliary buffer
		PFD_MAIN_PLANE,                 // main layer
		0,                              // reserved
		0, 0, 0                         // layer masks ignored
	};

	// Get the id number for the best match supported by the hardware device context
	// to what is described in pfd
	int pixelFormat = ChoosePixelFormat(m_pDC->GetSafeHdc(), &pfd);

	//If there's no match, report an error
	if (0 == pixelFormat)
	{
		AfxMessageBox(CString("ChoosePixelFormat failed"));
		return FALSE;
	}

	//If there is an acceptable match, set it as the current 
	if (FALSE == SetPixelFormat(m_pDC->GetSafeHdc(), pixelFormat, &pfd))
	{
		AfxMessageBox(CString("SetPixelFormat failed"));
		return FALSE;
	}

	//Create a context with this pixel format
	if (0 == (m_hRC = wglCreateContext(m_pDC->GetSafeHdc())))
	{
		AfxMessageBox(CString("wglCreateContext failed"));
		return FALSE;
	}

	//Make it current.
	if (FALSE == wglMakeCurrent(m_pDC->GetSafeHdc(), m_hRC))
	{
		AfxMessageBox(CString("wglMakeCurrent failed"));
		return FALSE;
	}

	return TRUE;
}

glm::ivec2 CAppDlg::ConvertMousePos(const CPoint& pos) const
{
	CRect wndRect;
	GetWindowRect(wndRect);
	return glm::ivec2{ pos.x,wndRect.Height() - pos.y };
}

void CAppDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	SetCapture();

	m_pMPSCore->LMouseDown(static_cast<mevent::Flag>(nFlags), ConvertMousePos(point));
	CDialogEx::OnLButtonDown(nFlags, point);
}

void CAppDlg::OnRButtonDown(UINT nFlags, CPoint point)
{
	SetCapture();

	m_pMPSCore->RMouseDown(static_cast<mevent::Flag>(nFlags), ConvertMousePos(point));
	CDialogEx::OnRButtonDown(nFlags, point);
}

void CAppDlg::OnMButtonDown(UINT nFlags, CPoint point)
{
	SetCapture();

	m_pMPSCore->WMouseDown(static_cast<mevent::Flag>(nFlags), ConvertMousePos(point));
	CDialogEx::OnMButtonDown(nFlags, point);
}

void CAppDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	if (GetCapture() == this)
	{
		ReleaseCapture();
	}

	m_pMPSCore->LMouseUp(static_cast<mevent::Flag>(nFlags), ConvertMousePos(point));
	CDialogEx::OnLButtonUp(nFlags, point);
}

void CAppDlg::OnRButtonUp(UINT nFlags, CPoint point)
{
	if (GetCapture() == this)
	{
		ReleaseCapture();
	}

	m_pMPSCore->RMouseUp(static_cast<mevent::Flag>(nFlags), ConvertMousePos(point));
	CDialogEx::OnRButtonUp(nFlags, point);
}

void CAppDlg::OnMButtonUp(UINT nFlags, CPoint point)
{
	if (GetCapture() == this)
	{
		ReleaseCapture();
	}

	m_pMPSCore->WMouseUp(static_cast<mevent::Flag>(nFlags), ConvertMousePos(point));
	CDialogEx::OnMButtonUp(nFlags, point);
}

void CAppDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	m_pMPSCore->MouseMove(static_cast<mevent::Flag>(nFlags), ConvertMousePos(point));
	CDialogEx::OnMouseMove(nFlags, point);
}

BOOL CAppDlg::OnMouseWheel(UINT nFlags, short zDelta, CPoint point)
{
	m_pMPSCore->MouseWheel(static_cast<mevent::Flag>(nFlags), { 0, static_cast<int>(zDelta / abs(zDelta)) }, ConvertMousePos(point));
	return CDialogEx::OnMouseWheel(nFlags, zDelta, point);
}

void CAppDlg::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	m_pMPSCore->KeyDown(nChar, nRepCnt, static_cast<mevent::Flag>(nFlags));
}

void CAppDlg::OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	m_pMPSCore->KeyUp(nChar, nRepCnt, static_cast<mevent::Flag>(nFlags));
}

BOOL CAppDlg::PreTranslateMessage(MSG* pMsg)
{
	switch (pMsg->message)
	{
	case WM_KEYDOWN:
		OnKeyDown(static_cast<UINT>(pMsg->wParam), LOWORD(pMsg->lParam), HIWORD(pMsg->lParam));
		break;
	case WM_KEYUP:
		OnKeyUp(static_cast<UINT>(pMsg->wParam), LOWORD(pMsg->lParam), HIWORD(pMsg->lParam));
		break;
	}
	return CDialogEx::PreTranslateMessage(pMsg);
}