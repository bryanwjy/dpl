// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/all_bits.h"
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/zero.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void add(...) noexcept = delete;
void sub(...) noexcept = delete;
void mul(...) noexcept = delete;
void div(...) noexcept = delete;

template <typename Result, typename L, typename R>
concept arithmetic_result =
    equivalent_simd_as<Result, common_simd_t<common_arithmetic_type, L, R>> &&
    common_arithmetic_with<Result, typename L::value_type> &&
    common_arithmetic_with<Result, typename R::value_type>;

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
