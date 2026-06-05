// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/masked_op.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/mantissa_width.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/type_traits/representation.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void isnans(...) noexcept = delete;

struct isnans_t;

template <typename T>
concept unqualified_canonical_isnans = requires(T arg) {
    { isnans(internal::abi<T>, arg) } -> exact_mask_for<T>;
};

template <typename T>
concept unqualified_extended_isnans = requires(T arg) {
    { isnans(arg) } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T>
concept expression_isnans =
    simd_expression<T> && invocable<isnans_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_isnans =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<isnans_t, canonical_type_t<T>>;

template <typename T>
concept extended_isnans = unqualified_extended_isnans<T> ||
    expression_isnans<T> || decayable_isnans<T>;

struct isnans_t : mx::masked_predicate<isnans_t> {
private:
    friend mx::masked_predicate<isnans_t>;

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg) noexcept {
        using uint = unsigned_representation_t<E>;
        using sint = signed_representation_t<E>;
        constexpr auto shift = sizeof(E) * char_bit_v - dx::mantissa_width_v<E>;
        return dx::cmpgt(
            dx::bwshift_left(dx::reinterpret<sint>(arg), imm<shift>), dx::zero);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_predicate<isnans_t, M, T> &&
        requires(M mask, T val) { isnans(internal::abi<T>, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return isnans(internal::abi<T>, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_predicate<isnans_t, M, T> &&
        requires(M mask, T val) { isnans(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return isnans(mask, val);
    }

public:
    template <simd_abi A, simd_floating_point_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        if constexpr (unqualified_canonical_isnans<basic_vector<E, A>>) {
            if consteval {
                return fallback(arg);
            } else {
                return isnans(internal::abi<A>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_isnans<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        return isnans(internal::abi<A>, arg);
    }

    template <extended_vector T>
    requires extended_isnans<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_extended_isnans<T>) {
            return isnans(arg);
        } else if constexpr (expression_isnans<T>) {
            return operator()(dx::evaluate(arg));
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    using mx::masked_predicate<isnans_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::isnans_t isnans{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
