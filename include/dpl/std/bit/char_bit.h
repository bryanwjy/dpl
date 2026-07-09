// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/popcount.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT inline constexpr size_t char_bit_v =
    __DPL popcount(static_cast<unsigned char>(-1));

DPL_DEFAULT_NAMESPACE_END
