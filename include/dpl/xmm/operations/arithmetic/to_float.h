// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/xmm/basic/abi.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {
template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To> element_cast(vector<ext::bfloat16> arg) noexcept;
template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To> element_cast(vector<ext::float16> arg) noexcept;
template <same_as<ext::bfloat16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To> element_cast(vector<float> arg) noexcept;
template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To> element_cast(vector<float> arg) noexcept;
} // namespace datapar::xmm
__DPL_DEFAULT_NAMESPACE_END
