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
void isfinite(...) noexcept = delete;

struct DPL_EMPTY_BASES isfinite_t :
    public math_operation_base<isfinite_t>,
    public maskable_predicate_base<isfinite_t> {
    using math_operation_base<isfinite_t>::operator();
    using maskable_predicate_base<isfinite_t>::operator();
};

template <>
struct operation_signature<isfinite_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct fallback_impl<isfinite_t> {
public:
    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(T val) noexcept {
        auto const inf = dx::broadcast<T>(dx::infinity);
        return dx::cmpneq(dx::bwand(val, inf), inf);
    }

    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(simd_mask_type_t<T> mask, T val) noexcept {
        auto const inf = dx::broadcast<T>(dx::infinity);
        return dx::cmpneq(mask, dx::bwand(val, inf), inf);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(M cmask, T val) noexcept {
        auto const inf = dx::broadcast<T>(dx::infinity);
        return dx::cmpneq(cmask, dx::bwand(val, inf), inf);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_isfinite = requires {
    {
        isfinite(internal::abi<A>, internal::declarg<T>())
    } -> same_as<simd_mask_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_canonical_misfinite = cpo_invocable<isfinite_t, T> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<isfinite_t, T>>) && requires {
        {
            isfinite(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<isfinite_t, T>>;
    };

template <>
struct canonical_impl<isfinite_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_isfinite<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(T arg) noexcept {
        return isfinite(internal::abi<T>, arg);
    }

    template <canonical_vector T>
    requires unqualified_canonical_misfinite<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(
        simd_mask_type_t<T> src, T val) noexcept {
        return isfinite(internal::abi<T>, src, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_misfinite<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(M cmask, T val) noexcept {
        return isfinite(internal::abi<T>, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T>
concept unqualified_extended_isfinite = requires {
    {
        isfinite(internal::declarg<T>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_extended_misfinite = cpo_invocable<isfinite_t, T> &&
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<isfinite_t, T>>) &&
    requires {
        {
            isfinite(internal::declarg<S>(), internal::declarg<T>())
        } -> equivalent_mask_with<cpo_result_t<isfinite_t, T>>;
    };

template <>
struct extended_impl<isfinite_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_isfinite<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) {
        return isfinite(__DPL forward<T>(arg));
    }

    template <simd_mask S, simd_vector T>
    requires (extended_mask<S> || extended_vector<T>) &&
        unqualified_extended_misfinite<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, T&& arg) {
        return isfinite(__DPL forward<S>(src), __DPL forward<T>(arg));
    }

    template <extended_vector T, result_cmask_for<isfinite_t, T> M>
    requires unqualified_extended_misfinite<
        launder_cmask_t<cpo_result_t<isfinite_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, T&& arg) {
        return isfinite(dx::to_const_mask<cpo_result_t<isfinite_t, T>>(cmask),
            __DPL forward<T>(arg));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::isfinite_t isfinite{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
