#include "stdafx.h"
#include "MPSSystem.h"

#include "../MPS_Object/MPSGBArchiver.h"

#include "../MPS_Render/MPSRenderManager.h"
#include "../MPS_Render/MPSCamera.h"

#include "../MPS_Object/MPSSPHModel.h"
#include "../MPS_Object/MPSSPHObject.h"
#include "../MPS_Object/MPSSPHMaterial.h"
#include "../MPS_Object/MPSSpatialHash.h"

mps::System::System() : SystemEventHandler{}, m_frame{ 0 }, b_runSim{ false }
{}

void mps::System::Initalize()
{
    SystemEventHandler::Initalize();


    {
        m_pGBArchiver = std::make_shared<mps::GBArchiver>();
        m_pGBArchiver->Initalize();
        m_pGBArchiver->UpdateLight(mps::LightParam{ glm::dvec3{ 10.0, 10.0, 10.0 }, glm::fvec4{ 1.0f, 1.0f, 1.0f, 1.0f } });
        m_pGBArchiver->m_physicsParam.gravity = { 0.0, -9.8, 0.0 };
        //m_pGBArchiver->m_physicsParam.gravity = REAL3{ 0.0 };
        m_pGBArchiver->m_physicsParam.dt = 0.005;
        m_pGBArchiver->m_physicsParam.min = { -5.0, -5.0, -5.0 };
        m_pGBArchiver->m_physicsParam.max = { 5.0, 5.0, 5.0 };
    }
    {
        m_pRenderManager = std::make_shared<mps::rndr::RenderManager>();
        m_pRenderManager->Initalize(m_pGBArchiver.get());

        m_pCamera = std::make_shared<mps::rndr::Camera>();
        m_pCameraHandler = std::make_unique<mps::rndr::CameraUserInputEventHandler>(m_pCamera.get());

        m_pSPHModel = std::make_shared<mps::SPHModel>(
            std::make_unique<mps::SPHMaterial>(),
            std::make_unique<mps::SPHObject>(),
            std::make_unique<mps::SpatialHash>());
        //ResizeParticle(5);
        ViscosityTestScene(5, 300);
        m_pRenderManager->AddModel(m_pSPHModel);
    }

    glEnable(GL_DEPTH_TEST);
}

void mps::System::OnWMouseDown(mevent::Flag flag, glm::ivec2 curPos)
{
    m_pCameraHandler->OnWMouseDown(flag, curPos);
}

void mps::System::OnWMouseUp(mevent::Flag flag, glm::ivec2 curPos)
{
    m_pCameraHandler->OnWMouseUp(flag, curPos);
}

void mps::System::OnMouseMove(mevent::Flag flag, glm::ivec2 curPos)
{
    m_pCameraHandler->OnMouseMove(flag, curPos);
}

void mps::System::OnMouseWheel(mevent::Flag flag, glm::ivec2 offset, glm::ivec2 curPos)
{
    m_pCameraHandler->OnMouseWheel(flag, offset, curPos);
}

void mps::System::OnKeyDown(uint32_t key, uint32_t repCnt, mevent::Flag flag)
{
    auto pSPHObject = static_cast<mps::SPHObject*>(m_pSPHModel->GetTarget());
    uint32_t sqNumParticles = static_cast<uint32_t>(powf(static_cast<float>(pSPHObject->GetSize()), 1.0f / 3.0f));
    uint32_t offset = 5;
    switch (key)
    {
    case 32: // ' ':
        b_runSim = !b_runSim;
        break;
    case 219: // '[':
        ResizeParticle((sqNumParticles > offset) ? sqNumParticles - offset : 1);
        break;
    case 221: // ']':
        ResizeParticle(sqNumParticles + offset);
        break;
    case 'g':
    case 'G':
    {
        if (m_pGBArchiver->m_physicsParam.gravity == REAL3{ 0.0 })
        {
            m_pGBArchiver->m_physicsParam.gravity = { 0.0, -9.8, 0.0 };
            OutputDebugStringA("Gravity On\n");
        }
        else
        {
            m_pGBArchiver->m_physicsParam.gravity = REAL3{ 0.0 };
            OutputDebugStringA("Gravity Off\n");
        }
        break;
    }
    case 'c':
    case 'C':
    {
        std::stringstream ss;
        ss << "Camera TransMatrix : ";

        const auto PrintVec = [&ss](const glm::fvec3& vec, bool bEnd)
        {
            ss << "{ " << vec[0] << ", " << vec[1] << ", " << vec[2];
            if (bEnd)   ss << " }" << std::endl;
            else        ss << " }, ";
        };
        PrintVec(m_pCamera->GetTransform()->GetPosition(), false);
        PrintVec(m_pCamera->GetTransform()->GetXDir(), false);
        PrintVec(m_pCamera->GetTransform()->GetYDir(), false);
        PrintVec(m_pCamera->GetTransform()->GetZDir(), true);
        OutputDebugStringA(ss.str().c_str());
        break;
    }
    default:
        break;
    }
}

void mps::System::OnKeyUp(uint32_t key, uint32_t repCnt, mevent::Flag flag)
{
    switch (key)
    {
    case 'A':
        break;
    case 'S':
        break;
    default:
        break;
    }
}

void mps::System::OnResize(int width, int height)
{
    m_pCamera->GetProjection()->SetAspectRatio(width, height);
}

#include "../MPS_Computer/AdvectUtil.h"
#include "../MPS_Computer/SPHUtil.h"
void mps::System::OnUpdate()
{
    if (!b_runSim) return;

    auto pSPHObject = static_cast<mps::SPHObject*>(m_pSPHModel->GetTarget());
    auto pSPHMaterial = static_cast<mps::SPHMaterial*>(m_pSPHModel->GetMaterial());
    auto pSpatialHash = static_cast<mps::SpatialHash*>(m_pSPHModel->GetTree());

    const auto sphRes = pSPHObject->GetDeviceResource<mps::SPHResource>();
    const auto pSPHParam = sphRes->GetSPHParam().lock();
    if (!pSPHParam) return;

    const auto& sphMaterialParam = pSPHMaterial->GetParam();
    const auto& hashParam = pSpatialHash->GetParam();
    const auto& physParam = m_pGBArchiver->m_physicsParam;
    pSpatialHash->UpdateHash(*pSPHParam);
    
    mps::kernel::sph::ComputeDensity(*pSPHParam, sphMaterialParam, hashParam);
    mps::kernel::sph::ComputeDFSPHFactor(*pSPHParam, sphMaterialParam, hashParam);
    //mps::kernel::sph::DensityColorTest(*pSPHParam, sphMaterialParam);

    mps::kernel::sph::ComputeDFSPHDivergenceFree(physParam, *pSPHParam, sphMaterialParam, hashParam);

    mps::kernel::ResetForce(*pSPHParam);
    mps::kernel::ApplyGravity(physParam, *pSPHParam);
    //mps::kernel::sph::ApplyExplicitSurfaceTension(*pSPHParam, sphMaterialParam, hashParam);
    mps::kernel::UpdateVelocity(physParam, *pSPHParam);

    /*mps::kernel::ResetForce(*pSPHParam);
    mps::kernel::sph::ApplyExplicitViscosity(*pSPHParam, sphMaterialParam, hashParam);
    mps::kernel::UpdateVelocity(physParam, *pSPHParam);*/

    //mps::kernel::sph::ApplyImplicitViscosity(physParam, *pSPHParam, sphMaterialParam, hashParam);
    //mps::kernel::sph::ApplyImplicitGDViscosity(physParam, *pSPHParam, sphMaterialParam, hashParam);
    mps::kernel::sph::ApplyImplicitCGViscosity(physParam, *pSPHParam, sphMaterialParam, hashParam);

    mps::kernel::sph::ComputeDFSPHConstantDensity(physParam, *pSPHParam, sphMaterialParam, hashParam);

    mps::kernel::BoundaryCollision(physParam, *pSPHParam);
    mps::kernel::UpdatePosition(physParam, *pSPHParam);
    
    m_pCamera->GetTransform()->Set({ 6.04415, 9.72579, -10.5085 }, { 0.86881, 0, 0.495144 }, { -0.217152, 0.898697, 0.381028 }, { -0.444984, -0.438562, 0.780798 });
    Capture(300, 4);
    m_frame++;
}

void mps::System::OnDraw()
{
    m_pCamera->Update(m_pGBArchiver.get());
    m_pGBArchiver->UpdateLightPos(m_pCamera->GetTransform()->GetPosition());

    glClearColor(0.9f, 0.99f, 0.96f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_pRenderManager->Draw(m_pGBArchiver.get());
}

void mps::System::Capture(uint32_t endFrame, uint32_t step)
{
    GLint m_viewport[4];
    glGetIntegerv(GL_VIEWPORT, m_viewport);
    const auto width = m_viewport[2] - m_viewport[0];
    const auto height = m_viewport[3] - m_viewport[1];

    if (m_frame == 0 || m_frame % step == 0)
    {
        static int index = 0;
        if (index == 0)
        {
            index++;
            return;
        }
        char filename[100];
        sprintf_s(filename, "../../capture/capture-%d.bmp", index);
        BITMAPFILEHEADER bf;
        BITMAPINFOHEADER bi;
        unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
        FILE* file;
        fopen_s(&file, filename, "wb");
        if (image != NULL)
        {
            if (file != NULL)
            {
                glReadPixels(0, 0, width, height, 0x80E0, GL_UNSIGNED_BYTE, image);
                memset(&bf, 0, sizeof(bf));
                memset(&bi, 0, sizeof(bi));
                bf.bfType = 'MB';
                bf.bfSize = sizeof(bf) + sizeof(bi) + width * height * 3;
                bf.bfOffBits = sizeof(bf) + sizeof(bi);
                bi.biSize = sizeof(bi);
                bi.biWidth = width;
                bi.biHeight = height;
                bi.biPlanes = 1;
                bi.biBitCount = 24;
                bi.biSizeImage = width * height * 3;
                fwrite(&bf, sizeof(bf), 1, file);
                fwrite(&bi, sizeof(bi), 1, file);
                fwrite(image, sizeof(unsigned char), width * height * 3, file);
                fclose(file);
            }
            free(image);
        }
        if (index == endFrame)
        {
            exit(0);
        }
        index++;
    }
}

void mps::System::ResizeParticle(size_t size)
{
    auto pSPHObject = static_cast<mps::SPHObject*>(m_pSPHModel->GetTarget());
    auto pSPHMaterial = static_cast<mps::SPHMaterial*>(m_pSPHModel->GetMaterial());
    auto pSpatialHash = static_cast<mps::SpatialHash*>(m_pSPHModel->GetTree());
    if (pSPHObject->GetSize() == size * size * size)
        return;

    MTimer::Start("ResizeParticle");

    REAL radius = 0.1f;// pSPHObject->GetSize() == 0 ? 0.1f : pSPHObject->m_radius.GetParam()[0];
    pSPHMaterial->SetRadius(radius * 4);

    REAL gap = radius * 0.8f;
    REAL offset = 0.001f;
    REAL stride = gap + gap + offset;
    REAL l = stride * (size - 1) + gap + gap;
    REAL startPos = -l * 0.5f;
    REAL mass = 1.0f;
    m_pGBArchiver->m_physicsParam.min =
    {
        startPos + gap - 1.5,
        startPos + gap - 1.5,
        startPos + gap - 1.5
    };
    m_pGBArchiver->m_physicsParam.max =
    {
        startPos + (size - 1) * stride + gap + 1.5,
        startPos + (size - 1) * stride + gap + 1.5,
        startPos + (size - 1) * stride + gap + 1.5
    };

    std::vector<REAL3> h_pos;
    std::vector<REAL> h_radius;
    std::vector<REAL3> h_vel(size * size * size, REAL3{ 1.0, 0.0, 0.0 });
    std::vector<REAL> h_mass(size * size * size, pSPHMaterial->GetMass());
    std::vector<glm::fvec4> h_color(size * size * size, glm::fvec4{ 0.0f, 0.0f, 1.0f, 1.0f });

    h_pos.reserve(size * size * size);
    h_radius.reserve(size * size * size);
    for (int z = 0; z < size; z++)
    {
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                REAL3 pos =
                {
                    startPos + x * stride + gap,
                    startPos + y * stride + gap,
                    startPos + z * stride + gap
                };
                h_pos.emplace_back(pos);
                h_radius.emplace_back(radius);
            }
        }
    }

    pSPHObject->Resize(h_radius.size());
    pSPHObject->m_pos.CopyFromHost(h_pos);
    pSPHObject->m_radius.CopyFromHost(h_radius);
    pSPHObject->m_velocity = h_vel;
    pSPHObject->m_mass = h_mass;
    pSPHObject->m_color.CopyFromHost(h_color);

    pSpatialHash->SetObjectSize(pSPHObject->GetSize());
    pSpatialHash->SetCeilSize(radius * 4);
    pSpatialHash->SetHashSize({ 64, 64, 64 });

    MTimer::End("ResizeParticle");
    {
        std::stringstream ss;
        ss << "Number Of Particle" << " : " << pSPHObject->GetSize() << std::endl;
        OutputDebugStringA(ss.str().c_str());
    }
}

void mps::System::ViscosityTestScene(size_t size, size_t height)
{
    auto pSPHObject = static_cast<mps::SPHObject*>(m_pSPHModel->GetTarget());
    auto pSPHMaterial = static_cast<mps::SPHMaterial*>(m_pSPHModel->GetMaterial());
    auto pSpatialHash = static_cast<mps::SpatialHash*>(m_pSPHModel->GetTree());

    const auto numParticle = size * size * height;
    if (pSPHObject->GetSize() == numParticle)
        return;

    MTimer::Start("ResizeParticle");

    REAL radius = 0.1f;
    pSPHMaterial->SetRadius(radius * 4);

    REAL gap = radius * 0.9f;
    REAL offset = 0.001f;
    REAL stride = gap + gap + offset;
    REAL lw = stride * (size - 1) + gap + gap;
    REAL lh = stride * (height - 1) + gap + gap;
    REAL startPos = -lw * 0.5f;
    REAL startH = 0.0f;
    REAL mass = 1.0f;
    m_pGBArchiver->m_physicsParam.min =
    {
        startPos + gap - 5.5,
        startH - 1.0,
        startPos + gap - 5.5
    };
    m_pGBArchiver->m_physicsParam.max =
    {
        startPos + (size - 1) * stride + gap + 5.5,
        startH - (height - 1) * stride + gap + 5.5,
        startPos + (size - 1) * stride + gap + 5.5
    };

    std::vector<REAL3> h_pos;
    std::vector<REAL> h_radius;
    std::vector<REAL3> h_vel(numParticle, REAL3{ 0.0, 0.0, 0.0 });
    std::vector<REAL> h_mass(numParticle, pSPHMaterial->GetMass());
    std::vector<glm::fvec4> h_color(numParticle, glm::fvec4{ 0.2f, 0.7f, 1.0f, 1.0f });

    h_pos.reserve(numParticle);
    h_radius.reserve(numParticle);
    for (int y = 0; y < height; y++)
    {
        for (int z = 0; z < size; z++)
        {
            for (int x = 0; x < size; x++)
            {
                REAL3 pos =
                {
                    startPos + x * stride + gap,
                    startPos + y * stride + gap,
                    startPos + z * stride + gap
                };
                h_pos.emplace_back(pos);
                h_radius.emplace_back(radius);
            }
        }
    }

    pSPHObject->Resize(h_radius.size());
    pSPHObject->m_pos.CopyFromHost(h_pos);
    pSPHObject->m_radius.CopyFromHost(h_radius);
    pSPHObject->m_velocity = h_vel;
    pSPHObject->m_mass = h_mass;
    pSPHObject->m_color.CopyFromHost(h_color);

    pSpatialHash->SetObjectSize(pSPHObject->GetSize());
    pSpatialHash->SetCeilSize(radius * 4);
    pSpatialHash->SetHashSize({ 64, 64, 64 });

    MTimer::End("ResizeParticle");
    {
        std::stringstream ss;
        ss << "Number Of Particle" << " : " << pSPHObject->GetSize() << std::endl;
        OutputDebugStringA(ss.str().c_str());
    }
}
