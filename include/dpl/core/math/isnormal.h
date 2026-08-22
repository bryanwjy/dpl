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
#  include "dpl/core/immediate/constants/infinity.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void isnormal(...) noexcept = delete;

struct DPL_EMPTY_BASES isnormal_t :
    public math_operation_base<isnormal_t>,
    public maskable_predicate_base<isnormal_t> {
    using math_operation_base<isnormal_t>::operator();
    using maskable_predicate_base<isnormal_t>::operator();
};

template <>
struct operation_signature<isnormal_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct fallback_impl<isnormal_t> {
public:
    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(T val) noexcept {
        using E = simd_element_type_t<T>;
        using sint_t = signed_representation_t<E>;
        constexpr auto shift =
            __DPL countr_zero(floating_point_traits<E>::exponent_mask);
        constexpr sint_t limit = sint_t(1) << shift;

        auto const vabs = dx::abs(val);
        auto const vlimit = dx::broadcast<mx::exponent_vector_t<T>>(limit);
        return dx::cmplt(dx::reinterpret<sint_t>(vabs), vlimit);
    }

    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(simd_mask_type_t<T> mask, T val) noexcept {
        using E = simd_element_type_t<T>;
        using sint_t = signed_representation_t<E>;
        constexpr auto shift =
            __DPL countr_zero(floating_point_traits<E>::exponent_mask);
        constexpr sint_t limit = sint_t(1) << shift;

        auto const vabs = dx::abs(val);
        auto const vlimit = dx::broadcast<mx::exponent_vector_t<T>>(limit);
        return dx::cmplt(mask, dx::reinterpret<sint_t>(vabs), vlimit);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(M cmask, T val) noexcept {
        using E = simd_element_type_t<T>;
        using sint_t = signed_representation_t<E>;
        constexpr auto shift =
            __DPL countr_zero(floating_point_traits<E>::exponent_mask);
        constexpr sint_t limit = sint_t(1) << shift;

        auto const vabs = dx::abs(val);
        auto const vlimit = dx::broadcast<mx::exponent_vector_t<T>>(limit);
        return dx::cmplt(cmask, dx::reinterpret<sint_t>(vabs), vlimit);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_isnormal = requires {
    {
        isnormal(internal::abi<A>, internal::declarg<T>())
    } -> same_as<simd_mask_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_canonical_misnormal = cpo_invocable<isnormal_t, T> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<isnormal_t, T>>) && requires {
        {
            isnormal(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<isnormal_t, T>>;
    };

template <>
struct canonical_impl<isnormal_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_isnormal<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(T arg) noexcept {
        return isnormal(internal::abi<T>, arg);
    }

    template <canonical_vector T>
    requires unqualified_canonical_misnormal<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(
        simd_mask_type_t<T> src, T val) noexcept {
        return isnormal(internal::abi<T>, src, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_misnormal<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(M cmask, T val) noexcept {
        return isnormal(internal::abi<T>, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T>
concept unqualified_extended_isnormal = requires {
    {
        isnormal(internal::declarg<T>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_extended_misnormal = cpo_invocable<isnormal_t, T> &&
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<isnormal_t, T>>) &&
    requires {
        {
            isnormal(internal::declarg<S>(), internal::declarg<T>())
        } -> equivalent_mask_with<cpo_result_t<isnormal_t, T>>;
    };

template <>
struct extended_impl<isnormal_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_isnormal<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) {
        return isnormal(__DPL forward<T>(arg));
    }

    template <simd_mask S, simd_vector T>
    requires (extended_mask<S> || extended_vector<T>) &&
        unqualified_extended_misnormal<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, T&& arg) {
        return isnormal(__DPL forward<S>(src), __DPL forward<T>(arg));
    }

    template <extended_vector T, result_cmask_for<isnormal_t, T> M>
    requires unqualified_extended_misnormal<
        launder_cmask_t<cpo_result_t<isnormal_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, T&& arg) {
        return isnormal(dx::to_const_mask<cpo_result_t<isnormal_t, T>>(cmask),
            __DPL forward<T>(arg));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::isnormal_t isnormal{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
