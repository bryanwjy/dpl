// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/element_cast/fallback.h" // IWYU pragma: export

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
