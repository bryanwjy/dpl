// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_size_with.h"
#include "dpl/core/concepts/simd_element.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace internal {
template <typename T>
using bit_for_t DPL_NODEBUG = typename bit_type<sizeof(T) * char_bit_v>::type;
} // namespace internal

namespace atom {
template <typename A, typename B>
concept common_bits_with = same_as<A, B> || requires(A a, B b) {
    typename internal::bit_for_t<A>;
    typename internal::bit_for_t<B>;
    requires same_as<internal::bit_for_t<A>, internal::bit_for_t<B>>;
    __DPL bit_cast<B>(a);
    __DPL bit_cast<A>(b);
    __DPL bit_cast<internal::bit_for_t<A>>(a);
    __DPL bit_cast<internal::bit_for_t<B>>(b);
};
} // namespace atom

DPL_EXPORT template <typename A, typename B>
concept common_bits_with =
    common_size_with<A, B> && atom::common_bits_with<A, B>;

DPL_EXPORT template <typename A, typename B>
concept common_bits_simd_with = common_class_with<A, B> &&
    common_bits_with<simd_lane_type_t<A>, simd_lane_type_t<B>>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
