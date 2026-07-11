// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/constructible_from.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept default_initializable = constructible_from<T> && requires {
    T{};
    ::new T;
};

__DPL_DEFAULT_NAMESPACE_END
