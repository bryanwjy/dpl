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

void isinf(...) noexcept = delete;

template <typename T>
concept unqualified_canonical_isinf = requires(T arg) {
    { isinf(internal::abi<T>, arg) } -> exact_mask_for<T>;
};

template <typename T>
concept unqualified_extended_isinf = requires(T arg) {
    { isinf(arg) } -> compatible_mask_with<T>;
};

struct isinf_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg) noexcept {
        return dx::cmpeq(dx::bwand(arg, dx::value_bits), dx::infinity);
    }

public:
    template <fixed_width_abi A, simd_floating_point_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        if constexpr (unqualified_canonical_isinf<basic_vector<E, A>>) {
            if consteval {
                return fallback(arg);
            } else {
                return isinf(internal::abi<A>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <scalable_abi A, simd_floating_point_for<A> E>
    requires unqualified_canonical_isinf<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        return isinf(internal::abi<A>, arg);
    }

    template <extended_vector T>
    requires unqualified_extended_isinf<T> ||
        (decayable_vector_for<T, operation_category::lane_agnostic> &&
            regular_invocable<isinf_t, canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_extended_isinf<T>) {
            return isinf(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::isinf_t isinf{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
