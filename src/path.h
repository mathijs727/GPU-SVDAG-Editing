#pragma once
#include "array2d.h"
#include "cuda_math.h"
#include "typedefs.h"
#include "utils.h"

struct Path {
public:
    uint3 path;

    HOST_DEVICE Path() { }
    HOST_DEVICE Path(uint3 path)
        : path(path)
    {
    }
    HOST_DEVICE Path(uint32 x, uint32 y, uint32 z)
        : path(make_uint3(x, y, z))
    {
    }

    HOST_DEVICE void ascend(uint32 levels)
    {
        path.x >>= levels;
        path.y >>= levels;
        path.z >>= levels;
    }
    HOST_DEVICE void descend(uint8 child)
    {
        path.x <<= 1;
        path.y <<= 1;
        path.z <<= 1;
        path.x |= (child & 0x4u) >> 2;
        path.y |= (child & 0x2u) >> 1;
        path.z |= (child & 0x1u) >> 0;
    }

    HOST_DEVICE float3 as_position(uint32 extraShift = 0) const
    {
        return make_float3(
            float(path.x << extraShift),
            float(path.y << extraShift),
            float(path.z << extraShift));
    }
    HOST_DEVICE uint32_t mortonU32() const
    {
        return Utils::morton3D(path.x, path.y, path.z);
    }
    HOST_DEVICE uint64_t mortonU64() const
    {
        return Utils::morton3D_64(path.x, path.y, path.z);
    }

    // level: level of the child!
    HOST_DEVICE uint8 child_index(uint32 level, uint32 totalLevels) const
    {
        check(level <= totalLevels);
        return uint8(
            (((path.x >> (totalLevels - level) & 0x1) == 0) ? 0 : 4) | (((path.y >> (totalLevels - level) & 0x1) == 0) ? 0 : 2) | (((path.z >> (totalLevels - level) & 0x1) == 0) ? 0 : 1));
    }

    HOST_DEVICE bool is_null() const
    {
        return path.x == 0 && path.y == 0 && path.z == 0;
    }

    HOST_DEVICE bool operator<(const Path& rhs) const
    {
        if (path.x < rhs.path.x)
            return true;
        else if (path.x > rhs.path.x)
            return false;
        else {
            if (path.y < rhs.path.y)
                return true;
            else if (path.y > rhs.path.y)
                return false;
            else {
                if (path.z < rhs.path.z)
                    return true;
                else
                    return false;
            }
        }
    }
    HOST_DEVICE bool operator==(const Path& rhs) const
    {
        return path.x == rhs.path.x && path.y == rhs.path.y && path.z == rhs.path.z;
    }

public:
    DEVICE static Path load(uint32_t x, uint32_t y, const StaticArray2D<uint3>& surface)
    {
        Path path;
        path.path = surface.read(x, y);
        return path;
    }
    DEVICE void store(uint32_t x, uint32_t y, StaticArray2D<uint3>& surface) const
    {
        surface.write(x, y, path);
    }
};