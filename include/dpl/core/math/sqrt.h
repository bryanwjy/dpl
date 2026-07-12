// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/accuracy.h"
#include "dpl/core/math/details/ldexp.h"
#include "dpl/core/math/details/rsqrt2.h"
#include "dpl/core/math/frexp.h"
#include "dpl/core/math/isfinite.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h" // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void sqrt(...) noexcept = delete;

struct DPL_EMPTY_BASES sqrt_t :
    private math_operation_base<sqrt_t>,
    private maskable_transform_base<sqrt_t> {
    using math_operation_base<sqrt_t>::operator();
    using maskable_transform_base<sqrt_t>::operator();
};

template <>
struct operation_signature<sqrt_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_msqrt = cpo_invocable<sqrt_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<sqrt_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            sqrt(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<sqrt_t, T>>;
    };

template <>
struct canonical_impl<sqrt_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { sqrt(internal::abi<A>, val); }
    {
        return sqrt(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msqrt<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return sqrt(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msqrt<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return sqrt(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msqrt<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return sqrt(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msqrt<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return sqrt(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_sqrt = requires {
    { sqrt(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_msqrt = cpo_invocable<sqrt_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<sqrt_t, T>>) &&
    requires {
        {
            sqrt(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<sqrt_t, T>>;
    };

template <>
struct extended_impl<sqrt_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_sqrt<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return sqrt(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msqrt<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return sqrt( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_msqrt<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return sqrt( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<sqrt_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msqrt<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return sqrt(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <simd_vector T, result_cmask_for<sqrt_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_msqrt<dx::zero_t,
            launder_cmask_t<cpo_result_t<sqrt_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return sqrt(zero, dx::to_const_mask<cpo_result_t<sqrt_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<sqrt_t> {
public:
    // TODO: float16/bfloat16
    template <simd_abi A, simd_element_for<A> E>
    requires same_as<float, E> || same_as<double, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<E, A> val) noexcept {
        // 2 * sqrt(2)
        auto const two = dx::broadcast<E, A>(2.0);
        auto const sqrt8 = dx::broadcast<E, A>(2.8284271247461900976033774484);

        auto const decomp = dx::frexp(val);
        auto const remtwo = decomp.exp & dx::one;
        auto const ifodd = dx::select(remtwo == dx::zero, sqrt8, two);
        auto const sig =
            (ifodd * decomp.fr) * mx::rsqrt2(mx::accuracy::maximum, decomp.fr);
        auto const result = mx::ldexp(mx::compliance::unsafe, //
            sig, (decomp.exp - dx::one) >> imm<1>);

        return dx::select(dx::isfinite(val) && val != dx::zero,
            dx::select(val < dx::zero, dx::all_bits, result), val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::sqrt_t sqrt{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
