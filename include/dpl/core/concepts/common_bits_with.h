// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
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
using bit_for_t = typename bit_type<sizeof(T) * char_bit_v>::type;
}

DPL_EXPORT template <typename A, typename B>
concept common_bits_with = sizeof(A) == sizeof(B) &&
    requires {
        typename internal::bit_for_t<A>;
        typename internal::bit_for_t<B>;
    } && same_as<internal::bit_for_t<A>, internal::bit_for_t<B>> &&
    requires(A a, B b) {
        __DPL bit_cast<B>(a);
        __DPL bit_cast<A>(b);
        __DPL bit_cast<internal::bit_for_t<A>>(a);
        __DPL bit_cast<internal::bit_for_t<B>>(b);
    };

DPL_EXPORT template <typename A, typename B>
concept simd_common_bits_with = simd_common_abi_with<A, B> &&
    common_bits_with<simd_element_type_t<A>, simd_element_type_t<B>>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
