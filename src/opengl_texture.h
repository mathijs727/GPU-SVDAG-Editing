#pragma once
#include "image.h"
#include "move_only.h"
#include <GL/glew.h>

struct OpenGLTexture {
public:
    MoveDefault<GLuint, (GLuint)-1> texID;
    uint32_t width, height;

public:
    OpenGLTexture(const Image&);
    ~OpenGLTexture();
    DEFAULT_MOVE(OpenGLTexture);

    void imguiImage() const;
};