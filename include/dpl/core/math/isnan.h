// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/compatible_mask_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/infinity.h"
#  include "dpl/core/constants/value_bits.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void isnan(...) noexcept = delete;

template <typename T>
concept unqualified_canonical_isnan = requires(T arg) {
    { isnan(internal::abi<T>, arg) } -> exact_mask_for<T>;
};

template <typename T>
concept unqualified_extended_isnan = requires(T arg) {
    { isnan(arg) } -> compatible_mask_with<T>;
};

struct isnan_t {
private:
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
    requires unqualified_extended_isnan<T> ||
        (decayable_vector_for<T, operation_category::lane_agnostic> &&
            regular_invocable<isnan_t, canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_extended_isnan<T>) {
            return isnan(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::isnan_t isnan{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
