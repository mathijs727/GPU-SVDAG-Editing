#include "opengl_texture.h"
#include <imgui.h>

OpenGLTexture::OpenGLTexture(const Image& image)
    : width(image.width)
    , height(image.height)
{
    // Create texture.
    glCreateTextures(GL_TEXTURE_2D, 1, &texID);
    glTextureStorage2D(texID, 1, GL_RGBA8, image.width, image.height);
    // Upload image to GPU memory.
    glTextureSubImage2D(texID, 0, 0, 0, image.width, image.height, GL_RGBA, GL_UNSIGNED_BYTE, image.pixels.data());

    // Nearest neighbor sampling.
    glTextureParameteri(texID, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(texID, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

OpenGLTexture::~OpenGLTexture()
{
    if (texID != (GLuint)-1)
        glDeleteTextures(1, &texID);
}

void OpenGLTexture::imguiImage() const
{
    const float contentRegionWidth = ImGui::GetContentRegionAvail().x;
    const float aspect = (float)width / (float)height;
    ImGui::Image((void*)(intptr_t)((GLuint)texID), ImVec2(contentRegionWidth, contentRegionWidth / aspect));
}
