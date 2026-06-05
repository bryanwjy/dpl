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
void isfinite(...) noexcept = delete;

struct isfinite_t;

template <typename T>
concept unqualified_canonical_isfinite = requires(T arg) {
    { isfinite(internal::abi<T>, arg) } -> exact_mask_for<T>;
};

template <typename T>
concept unqualified_extended_isfinite = requires(T arg) {
    { isfinite(arg) } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T>
concept expression_isfinite =
    simd_expression<T> && invocable<isfinite_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_isfinite =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<isfinite_t, canonical_type_t<T>>;

template <typename T>
concept extended_isfinite = unqualified_extended_isfinite<T> ||
    expression_isfinite<T> || decayable_isfinite<T>;

struct isfinite_t : private mx::masked_predicate<isfinite_t> {
private:
    friend mx::masked_predicate<isfinite_t>;

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg) noexcept {
        return dx::cmpneq(dx::bwand(arg, dx::infinity), dx::infinity);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_predicate<isfinite_t, M, T> &&
        requires(M mask, T val) { isfinite(internal::abi<T>, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return isfinite(internal::abi<T>, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_predicate<isfinite_t, M, T> &&
        requires(M mask, T val) { isfinite(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return isfinite(mask, val);
    }

public:
    template <simd_abi A, simd_floating_point_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        if constexpr (unqualified_canonical_isfinite<basic_vector<E, A>>) {
            if consteval {
                return fallback(arg);
            } else {
                return isfinite(internal::abi<A>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_isfinite<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        return isfinite(internal::abi<A>, arg);
    }

    template <extended_vector T>
    requires extended_isfinite<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_extended_isfinite<T>) {
            return isfinite(arg);
        } else if constexpr (expression_isfinite<T>) {
            return operator()(dx::evaluate(arg));
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    using mx::masked_predicate<isfinite_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::isfinite_t isfinite{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
