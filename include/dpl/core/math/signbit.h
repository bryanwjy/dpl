// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/compatible_mask_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/type_traits/representation.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void signbit(...) noexcept = delete;

struct signbit_t;

template <typename T>
concept unqualified_canonical_signbit = requires(T arg) {
    { signbit(internal::abi<T>, arg) } -> exact_mask_for<T>;
};

template <typename T>
concept unqualified_extended_signbit = requires(T arg) {
    { signbit(arg) } -> extended_operation_mask<typename T::abi_type>;
};

template <typename T>
concept expression_signbit =
    simd_expression<T> && invocable<signbit_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_signbit =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<signbit_t, canonical_type_t<T>>;

template <typename T>
concept extended_signbit = unqualified_extended_signbit<T> ||
    expression_signbit<T> || decayable_signbit<T>;

struct signbit_t : private mx::masked_predicate<signbit_t> {
private:
    friend mx::masked_predicate<signbit_t>;

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg) noexcept {
        using sint = signed_representation_t<E>;
        return dx::reinterpret<E>(
            dx::cmplt(dx::reinterpret<sint>(arg), dx::zero));
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_predicate<signbit_t, M, T> &&
        requires(M mask, T val) { signbit(internal::abi<T>, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return signbit(internal::abi<T>, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_predicate<signbit_t, M, T> &&
        requires(M mask, T val) { signbit(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return signbit(mask, val);
    }

public:
    template <simd_abi A, simd_floating_point_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        if constexpr (unqualified_canonical_signbit<basic_vector<E, A>>) {
            if consteval {
                return fallback(arg);
            } else {
                return signbit(internal::abi<A>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_signbit<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        return signbit(internal::abi<A>, arg);
    }

    template <extended_vector T>
    requires extended_signbit<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_extended_signbit<T>) {
            return signbit(arg);
        } else if constexpr (expression_signbit<T>) {
            return operator()(dx::evaluate(arg));
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    using mx::masked_predicate<signbit_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::signbit_t signbit{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
