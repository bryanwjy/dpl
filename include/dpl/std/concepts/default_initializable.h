// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/constructible_from.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept default_initializable = constructible_from<T> && requires {
    T{};
    ::new T;
};

DPL_DEFAULT_NAMESPACE_END
