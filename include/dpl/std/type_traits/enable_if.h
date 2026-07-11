// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <bool, typename T>
struct enable_if {};

template <typename T>
struct enable_if<true, T> {
    using type DPL_NODEBUG = T;
};

template <bool V, typename T>
using enable_if_t DPL_NODEBUG = typename enable_if<V, T>::type;

__DPL_DEFAULT_NAMESPACE_END
