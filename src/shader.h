#pragma once

#include "typedefs.h"
#include <filesystem>

GLuint LoadShaders(const std::filesystem::path& vertexFilePath, const std::filesystem::path& fragmentFilePath);