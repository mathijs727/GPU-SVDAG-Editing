#pragma once

#define NO_COPY(Name)           \
    Name(const Name&) = delete; \
    Name& operator=(const Name&) = delete
#define NO_MOVE(Name)      \
    Name(Name&&) = delete; \
    Name& operator=(Name&&) = delete
#define DEFAULT_COPY(Name)       \
    Name(const Name&) = default; \
    Name& operator=(const Name&) = default
#define DEFAULT_MOVE(Name)  \
    Name(Name&&) = default; \
    Name& operator=(Name&&) = default

template <typename T, T DefaultValue>
struct MoveDefault {
    T value { DefaultValue };

    inline MoveDefault()
        : value(DefaultValue)
    {
    }
    inline MoveDefault(T v)
        : value(v)
    {
    }
    inline MoveDefault(MoveDefault<T, DefaultValue>&& other)
        : value(other.value)
    {
        other.value = DefaultValue;
    }
    inline MoveDefault<T, DefaultValue>& operator=(T v)
    {
        this->value = v;
        return *this;
    }
    inline MoveDefault<T, DefaultValue>& operator=(MoveDefault<T, DefaultValue>&& other)
    {
        value = other.value;
        other.value = DefaultValue;
        return *this;
    }
    inline operator bool() const
    {
        return static_cast<bool>(value);
    }

    inline T* operator&()
    {
        return &value;
    }
    inline const T* operator&() const
    {
        return &value;
    }
    inline operator T()
    {
        return value;
    }
    inline operator const T() const
    {
        return value;
    }
};

/*// Work-around for compilers that don't fully implement C++20 and don't support
// class types as non-type template parameters (DefaultValue in this case).
template <typename T>
struct MoveDefaultClass {
    const T DefaultValue;
    T value;

    inline MoveDefaultClass(const T& defaultValue)
        : DefaultValue(defaultValue)
    {
    }
    inline MoveDefaultClass(MoveDefaultClass&& other)
        : value(other.value)
    {
        other.value = DefaultValue;
    }
    inline MoveDefaultClass& operator=(T v)
    {
        this->value = v;
        return *this;
    }
    inline MoveDefaultClass& operator=(MoveDefaultClass&& other)
    {
        value = other.value;
        other.value = DefaultValue;
        return *this;
    }
    inline operator bool() const
    {
        return static_cast<bool>(value);
    }
    inline operator T()
    {
        return value;
    }
    inline operator const T() const
    {
        return value;
    }
};*/

template <typename T>
struct MovePointer {
    T* pointer { nullptr };

    MovePointer() = default;
    inline MovePointer(T* p)
        : pointer(p)
    {
    }
    inline MovePointer(MovePointer<T>&& other)
        : pointer(other.pointer)
    {
        other.pointer = nullptr;
    }
    inline MovePointer<T>& operator=(T* pOther)
    {
        assert(!pointer);
        pointer = pOther;
        return *this;
    }
    inline MovePointer<T>& operator=(MovePointer<T>&& other)
    {
        pointer = other.pointer;
        other.pointer = nullptr;
        return *this;
    }
    inline operator bool() const
    {
        return pointer != nullptr;
    }
    inline operator T*()
    {
        return pointer;
    }
    inline operator const T*() const
    {
        return pointer;
    }
    inline T* operator->()
    {
        return pointer;
    }
    inline const T* operator->() const
    {
        return pointer;
    }
};
