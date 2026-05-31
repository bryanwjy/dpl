// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fixup.h"
#include "dpl/core/math/frexp.h"
#include "dpl/core/math/internal/accuracy.h"
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/masked_op.h"
#include "dpl/core/math/internal/rsqrt2.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void rsqrt(...) noexcept = delete;

struct rsqrt_t;

template <typename T>
concept unqualified_canonical_rsqrt = requires(T val) {
    {
        rsqrt(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_rsqrt = requires(T val) {
    { rsqrt(val) } -> extended_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_rsqrt = unqualified_extended_rsqrt<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<rsqrt_t, canonical_type_t<T>>);

struct rsqrt_t : private mx::masked_operation<rsqrt_t> {
private:
    friend mx::masked_operation<rsqrt_t>;

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
        constexpr auto inv_sqrt2 = dx::broadcast<E, A>(mx::rsqrt2(dx::one));

        auto const decomp = dx::frexp(val);
        auto const remtwo = decomp.exp & dx::one;
        auto const reduced = mx::rsqrt2(mx::accuracy::speed, decomp.fr);
        auto result = mx::ldexp(mx::compliance::unsafe, reduced,
            -((decomp.exp - dx::one) >> imm<1>));
        result = dx::multiply(result, remtwo == dx::zero, result, inv_sqrt2);
        return dx::fixup(val, result,
            fpfix::condition<fpfix::nan, fpfix::revert>       //
                | fpfix::condition<fpfix::infinity, dx::zero> //
                | fpfix::condition<fpfix::negative, dx::nan>);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::maskable_operator<rsqrt_t, S, M, T> &&
        mx::canonical_operator_args<S, M, T> && requires(S src, M mask, T val) {
            rsqrt(internal::abi<T>, src, mask, val);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return rsqrt(internal::abi<T>, src, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::maskable_operator<rsqrt_t, S, M, T> &&
        (!mx::canonical_operator_args<S, M, T>) &&
        requires(S src, M mask, T val) { rsqrt(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return rsqrt(src, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::maskable_zoperator<rsqrt_t, M, T> &&
        mx::canonical_zoperator_args<rsqrt_t, M, T> && requires(M mask, T val) {
            rsqrt(internal::abi<T>, dx::zero, mask, val);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return rsqrt(internal::abi<T>, dx::zero, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::maskable_zoperator<rsqrt_t, M, T> &&
        (!mx::canonical_zoperator_args<rsqrt_t, M, T>) &&
        requires(M mask, T val) { rsqrt(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return rsqrt(dx::zero, mask, val);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_rsqrt<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return rsqrt(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_rsqrt<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return rsqrt(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires unqualified_rsqrt<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_rsqrt<T>) {
            return rsqrt(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    using mx::masked_operation<rsqrt_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::rsqrt_t rsqrt{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
