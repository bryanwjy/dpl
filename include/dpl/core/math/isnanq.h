// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/predicate.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/mantissa_width.h"
#  include "dpl/core/immediate/constants/msb.h"
#  include "dpl/core/immediate/constants/value_bits.h"
#  include "dpl/core/operations/arithmetic/abs.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/type_traits/representation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void isnanq(...) noexcept = delete;

struct DPL_EMPTY_BASES isnanq_t :
    public math_operation_base<isnanq_t>,
    public maskable_predicate_base<isnanq_t> {
    using math_operation_base<isnanq_t>::operator();
    using maskable_predicate_base<isnanq_t>::operator();
};

template <>
struct operation_signature<isnanq_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct fallback_impl<isnanq_t> {
public:
    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(T val) noexcept {
        using E = simd_element_type_t<T>;
        using sint_t = signed_representation_t<E>;
        constexpr sint_t signaling_bit = sint_t(1)
            << (dx::mantissa_width_v<E> - 1);
        constexpr sint_t max_snan =
            __DPL bit_cast<sint_t>(dx::value_bits_v<E>) ^ signaling_bit;

        auto const abs_val = dx::bwandnot(val, dx::msb);
        return dx::cmpgt(dx::reinterpret<sint_t>(abs_val), max_snan);
    }

    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(simd_mask_type_t<T> mask, T val) noexcept {
        using E = simd_element_type_t<T>;
        using sint_t = signed_representation_t<E>;
        constexpr sint_t signaling_bit = sint_t(1)
            << (dx::mantissa_width_v<E> - 1);
        constexpr sint_t max_snan =
            __DPL bit_cast<sint_t>(dx::value_bits_v<E>) ^ signaling_bit;

        auto const abs_val = dx::bwandnot(val, dx::msb);
        auto const max = dx::broadcast<mx::exponent_vector_t<T>>(max_snan);
        return dx::cmpgt(mask, dx::reinterpret<sint_t>(abs_val), max);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(M cmask, T val) noexcept {
        using E = simd_element_type_t<T>;
        using sint_t = signed_representation_t<E>;
        constexpr sint_t signaling_bit = sint_t(1)
            << (dx::mantissa_width_v<E> - 1);
        constexpr sint_t max_snan =
            __DPL bit_cast<sint_t>(dx::value_bits_v<E>) ^ signaling_bit;

        auto const abs_val = dx::bwandnot(val, dx::msb);
        auto const max = dx::broadcast<mx::exponent_vector_t<T>>(max_snan);
        return dx::cmpgt(cmask, dx::reinterpret<sint_t>(abs_val), max);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_isnanq = requires {
    {
        isnanq(internal::abi<A>, internal::declarg<T>())
    } -> same_as<simd_mask_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_canonical_misnanq = cpo_invocable<isnanq_t, T> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<isnanq_t, T>>) && requires {
        {
            isnanq(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<isnanq_t, T>>;
    };

template <>
struct canonical_impl<isnanq_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_isnanq<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(T arg) noexcept {
        return isnanq(internal::abi<T>, arg);
    }

    template <canonical_vector T>
    requires unqualified_canonical_misnanq<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(
        simd_mask_type_t<T> src, T val) noexcept {
        return isnanq(internal::abi<T>, src, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_misnanq<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(M cmask, T val) noexcept {
        return isnanq(internal::abi<T>, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T>
concept unqualified_extended_isnanq = requires {
    {
        isnanq(internal::declarg<T>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_extended_misnanq = cpo_invocable<isnanq_t, T> &&
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<isnanq_t, T>>) &&
    requires {
        {
            isnanq(internal::declarg<S>(), internal::declarg<T>())
        } -> equivalent_mask_with<cpo_result_t<isnanq_t, T>>;
    };

template <>
struct extended_impl<isnanq_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_isnanq<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) {
        return isnanq(__DPL forward<T>(arg));
    }

    template <simd_mask S, simd_vector T>
    requires (extended_mask<S> || extended_vector<T>) &&
        unqualified_extended_misnanq<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, T&& arg) {
        return isnanq(__DPL forward<S>(src), __DPL forward<T>(arg));
    }

    template <extended_vector T, result_cmask_for<isnanq_t, T> M>
    requires unqualified_extended_misnanq<
        launder_cmask_t<cpo_result_t<isnanq_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, T&& arg) {
        return isnanq(dx::to_const_mask<cpo_result_t<isnanq_t, T>>(cmask),
            __DPL forward<T>(arg));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::isnanq_t isnanq{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
