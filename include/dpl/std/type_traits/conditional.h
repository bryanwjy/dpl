// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <bool, typename T, typename F>
struct conditional {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <typename T, typename F>
struct conditional<false, T, F> {
    using type DPL_NODEBUG = F;
};

DPL_EXPORT template <bool V, typename T, typename F>
using conditional_t DPL_NODEBUG = typename conditional<V, T, F>::type;

DPL_DEFAULT_NAMESPACE_END
