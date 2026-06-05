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
#  include "dpl/core/constants/value_bits.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void isnan(...) noexcept = delete;

struct isnan_t;

template <typename T>
concept unqualified_canonical_isnan = requires(T arg) {
    { isnan(internal::abi<T>, arg) } -> exact_mask_for<T>;
};

template <typename T>
concept unqualified_extended_isnan = requires(T arg) {
    { isnan(arg) } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T>
concept expression_isnan =
    simd_expression<T> && invocable<isnan_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_isnan =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<isnan_t, canonical_type_t<T>>;

template <typename T>
concept extended_isnan =
    unqualified_extended_isnan<T> || expression_isnan<T> || decayable_isnan<T>;

struct isnan_t : private mx::masked_predicate<isnan_t> {
private:
    friend mx::masked_predicate<isnan_t>;

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg) noexcept {
        using uint = unsigned_representation_t<E>;
        constexpr auto inf_bits =
            dx::reinterpret<uint>(dx::infinity_v<basic_vector<E, A>>);
        constexpr auto abs_bits =
            dx::reinterpret<uint>(dx::value_bits_v<basic_vector<E, A>>);
        auto const abs_val = dx::bwand(dx::reinterpret<uint>(arg), abs_bits);
        return dx::cmpgt(abs_val, inf_bits);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_predicate<isnan_t, M, T> &&
        requires(M mask, T val) { isnan(internal::abi<T>, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return isnan(internal::abi<T>, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_predicate<isnan_t, M, T> &&
        requires(M mask, T val) { isnan(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return isnan(mask, val);
    }

public:
    template <simd_abi A, simd_floating_point_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        if constexpr (unqualified_canonical_isnan<basic_vector<E, A>>) {
            if consteval {
                return fallback(arg);
            } else {
                return isnan(internal::abi<A>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_isnan<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        return isnan(internal::abi<A>, arg);
    }

    template <extended_vector T>
    requires extended_isnan<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_extended_isnan<T>) {
            return isnan(arg);
        } else if constexpr (expression_isnan<T>) {
            return operator()(dx::evaluate(arg));
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    using mx::masked_predicate<isnan_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::isnan_t isnan{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
