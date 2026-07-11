// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
struct type_identity {
    using type DPL_NODEBUG = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;

__DPL_DEFAULT_NAMESPACE_END
