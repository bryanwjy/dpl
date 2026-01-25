// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
using type_identity_t = T;

DPL_EXPORT template <typename T>
struct type_identity {
    using type DPL_NODEBUG = T;
};

DPL_DEFAULT_NAMESPACE_END
