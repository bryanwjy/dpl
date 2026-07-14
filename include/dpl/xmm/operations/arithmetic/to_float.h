// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/xmm/basic/abi.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {
template <typename>
void element_cast(...) noexcept = delete;

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<float> to_float(vector<ext::bfloat16> arg) noexcept;
} // namespace datapar::xmm
__DPL_DEFAULT_NAMESPACE_END
