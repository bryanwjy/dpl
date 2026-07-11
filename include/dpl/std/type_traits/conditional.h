// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <bool, typename T, typename F>
struct conditional {
    using type DPL_NODEBUG = T;
};

template <typename T, typename F>
struct conditional<false, T, F> {
    using type DPL_NODEBUG = F;
};

template <bool V, typename T, typename F>
using conditional_t DPL_NODEBUG = typename conditional<V, T, F>::type;

__DPL_DEFAULT_NAMESPACE_END
