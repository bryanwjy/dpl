// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include <stdint.h>
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

using ptrdiff_t = decltype((char*)0 - (char*)0);
using size_t = decltype(sizeof(0));

#if DPL_SUPPORTS_FLOAT16
using float16 = decltype(0.0f16);
#endif
#if DPL_SUPPORTS_FLOAT32
using float32 = decltype(0.0f32);
#endif
#if DPL_SUPPORTS_FLOAT64
using float64 = decltype(0.0f64);
#endif
#if DPL_SUPPORTS_FLOAT128
using float128 = decltype(0.0f128);
#endif
#if DPL_SUPPORTS_BFLOAT16
using bfloat16 = decltype(0.0bf16);
#endif

#if DPL_SUPPORTS_INT128
using int128 = __int128_t;
using uint128 = __uint128_t;
#endif

using int64 = int64_t;
using int32 = int32_t;
using int16 = int16_t;
using int8 = int8_t;
using uint64 = uint64_t;
using uint32 = uint32_t;
using uint16 = uint16_t;
using uint8 = uint8_t;

__DPL_DEFAULT_NAMESPACE_END
