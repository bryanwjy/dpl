// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/masked_op.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/infinity.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void isnormal(...) noexcept = delete;

struct isnormal_t;

template <typename T>
concept unqualified_canonical_isnormal = requires(T arg) {
    { isnormal(internal::abi<T>, arg) } -> exact_mask_for<T>;
};

template <typename T>
concept unqualified_extended_isnormal = requires(T arg) {
    { isnormal(arg) } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T>
concept expression_isnormal =
    simd_expression<T> && invocable<isnormal_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_isnormal =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<isnormal_t, canonical_type_t<T>>;

template <typename T>
concept extended_isnormal = unqualified_extended_isnormal<T> ||
    expression_isnormal<T> || decayable_isnormal<T>;

struct isnormal_t : private mx::masked_predicate<isnormal_t> {
private:
    friend mx::masked_predicate<isnormal_t>;

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg) noexcept {
        auto const val = dx::bwand(arg, dx::infinity);
        return val != dx::infinity && val > dx::zero;
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_predicate<isnormal_t, M, T> &&
        requires(M mask, T val) { isnormal(internal::abi<T>, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return isnormal(internal::abi<T>, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_predicate<isnormal_t, M, T> &&
        requires(M mask, T val) { isnormal(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return isnormal(mask, val);
    }

public:
    template <simd_abi A, simd_floating_point_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        if constexpr (unqualified_canonical_isnormal<basic_vector<E, A>>) {
            if consteval {
                return fallback(arg);
            } else {
                return isnormal(internal::abi<A>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_isnormal<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        return isnormal(internal::abi<A>, arg);
    }

    template <extended_vector T>
    requires extended_isnormal<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_extended_isnormal<T>) {
            return isnormal(arg);
        } else if constexpr (expression_isnormal<T>) {
            return operator()(dx::evaluate(arg));
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    using mx::masked_predicate<isnormal_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::isnormal_t isnormal{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
