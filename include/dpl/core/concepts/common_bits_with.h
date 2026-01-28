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

template <typename A, typename B>
concept common_bits_with = same_as<A, B> || (common_size_with<A, B> &&
    requires {
        typename internal::bit_for_t<A>;
        typename internal::bit_for_t<B>;
    } && same_as<internal::bit_for_t<A>, internal::bit_for_t<B>> &&
    requires(A a, B b) {
        __DPL bit_cast<B>(a);
        __DPL bit_cast<A>(b);
        __DPL bit_cast<internal::bit_for_t<A>>(a);
        __DPL bit_cast<internal::bit_for_t<B>>(b);
    });

template <typename T, typename U>
concept enum_bit_operable = (flag_enumeration<T> && same_as<T, U>) ||
    (unscoped_enumeration<T> &&
        (same_as<T, U> || (integral<U> && common_bits_with<T, U>)) &&
        requires(T lhs, U rhs) {
            requires !requires { operator&(lhs, rhs); } ||
                requires { lhs & rhs; };
            requires !requires { operator|(lhs, rhs); } ||
                requires { lhs | rhs; };
            requires !requires { operator^(lhs, rhs); } ||
                requires { lhs ^ rhs; };
            requires !requires { operator~(lhs); } || requires { ~lhs; };
        });

/**
 * An enum, T, is common bit with itself iff T statisfies flag_enumeration or T
 * is an unscoped enumeration that does not have any bitwise operations with
 * itself found via ADL.
 *
 * An unscoped enum, T, is common bit with an integral U, if they satisfies
 * internal::common_bits_with and no bitwise operations between T and U are
 * found via ADL
 */
template <typename T, typename U>
concept enum_common_bits_with = (enumeration<T> && enum_bit_operable<T, U>) ||
    (enumeration<U> && enum_bit_operable<U, T>);

} // namespace internal

DPL_EXPORT template <typename A, typename B>
concept common_bits_with =
    (!enumeration<A> && !enumeration<B> && internal::common_bits_with<A, B>) ||
    internal::enum_bit_operable<A, B>;

DPL_EXPORT template <typename A, typename B>
concept common_bits_simd_with = simd_common_abi_with<A, B> &&
    common_bits_with<simd_element_type_t<A>, simd_element_type_t<B>>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
