// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.numbers;
import :core.details.numbers;

__DPL_DEFAULT_NAMESPACE_BEGIN
// NOLINTBEGIN(misc-unused-using-decls,misc-unused-alias-decls)
using __DPL floating_point_like;
using __DPL floating_point_traits;
using __DPL integral_traits;
namespace ext {
using __DPL ext::bfloat16;
using __DPL ext::float16;
inline namespace literals {
using __DPL ext::literals::operator""_bf16;
using __DPL ext::literals::operator""_f16;
} // namespace literals
} // namespace ext

namespace ext_literals = ext::literals;
// NOLINTEND(misc-unused-using-decls,misc-unused-alias-decls)
__DPL_DEFAULT_NAMESPACE_END
