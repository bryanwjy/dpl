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
#  include "dpl/core/immediate/constants/value_bits.h"
#  include "dpl/core/operations/arithmetic/abs.h"
#  include "dpl/core/operations/compare.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void isinf(...) noexcept = delete;

struct DPL_EMPTY_BASES isinf_t :
    public math_operation_base<isinf_t>,
    public maskable_predicate_base<isinf_t> {
    using math_operation_base<isinf_t>::operator();
    using maskable_predicate_base<isinf_t>::operator();
};

template <>
struct fallback_impl<isinf_t> {
public:
    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(T val) noexcept {
        auto const inf = dx::broadcast<T>(dx::infinity);
        return dx::cmpeq(dx::abs(val), inf);
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(simd_mask_type_t<T> mask, T val) noexcept {
        auto const inf = dx::broadcast<T>(dx::infinity);
        return dx::cmpeq(mask, dx::abs(val), inf);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires floating_point_like<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(M cmask, T val) noexcept {
        auto const inf = dx::broadcast<T>(dx::infinity);
        return dx::cmpeq(cmask, dx::abs(val), inf);
    }
};

template <>
struct operation_signature<isinf_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_isinf = requires {
    {
        isinf(internal::abi<A>, internal::declarg<T>())
    } -> same_as<simd_mask_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_canonical_misinf = cpo_invocable<isinf_t, T> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<isinf_t, T>>) && requires {
        {
            isinf(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<isinf_t, T>>;
    };

template <>
struct canonical_impl<isinf_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_isinf<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(T arg) noexcept {
        return isinf(internal::abi<T>, arg);
    }

    template <canonical_vector T>
    requires unqualified_canonical_misinf<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(
        simd_mask_type_t<T> src, T val) noexcept {
        return isinf(internal::abi<T>, src, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_misinf<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<T> operator()(M cmask, T val) noexcept {
        return isinf(internal::abi<T>, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T>
concept unqualified_extended_isinf = requires {
    {
        isinf(internal::declarg<T>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_extended_misinf = cpo_invocable<isinf_t, T> &&
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<isinf_t, T>>) &&
    requires {
        {
            isinf(internal::declarg<S>(), internal::declarg<T>())
        } -> equivalent_mask_with<cpo_result_t<isinf_t, T>>;
    };

template <>
struct extended_impl<isinf_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_isinf<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) {
        return isinf(__DPL forward<T>(arg));
    }

    template <simd_mask S, simd_vector T>
    requires (extended_mask<S> || extended_vector<T>) &&
        unqualified_extended_misinf<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, T&& arg) {
        return isinf(__DPL forward<S>(src), __DPL forward<T>(arg));
    }

    template <extended_vector T, result_cmask_for<isinf_t, T> M>
    requires unqualified_extended_misinf<
        launder_cmask_t<cpo_result_t<isinf_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, T&& arg) {
        return isinf(dx::to_const_mask<cpo_result_t<isinf_t, T>>(cmask),
            __DPL forward<T>(arg));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::isinf_t isinf{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
