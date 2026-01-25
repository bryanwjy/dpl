// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <bool, typename T>
struct enable_if {};

DPL_EXPORT template <typename T>
struct enable_if<true, T> {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <bool V, typename T>
using enable_if_t DPL_NODEBUG = typename enable_if<V, T>::type;

DPL_DEFAULT_NAMESPACE_END
