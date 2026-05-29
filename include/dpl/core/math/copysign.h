// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void copysign(...) noexcept = delete;

struct copysign_t;

template <typename L, typename R = L, typename A = common_abi_t<L, R>>
concept unqualified_canonical_copysign = requires(L lhs, R rhs) {
    {
        copysign(internal::abi<A>, lhs, rhs)
    } -> canonical_arithmetic_result<L, L, A>;
};

template <typename L, typename R = L, typename A = common_abi_t<L, R>>
concept unqualified_extended_copysign = requires(L lhs, R rhs) {
    { copysign(lhs, rhs) } -> extended_arithmetic_result<L, L, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_copysign = unqualified_extended_copysign<L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        regular_invocable<copysign_t, canonical_type_t<L>,
            canonical_type_t<R>>);

struct copysign_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(
            basic_vector<E, A> magnitude, basic_vector<E, A> sign) noexcept {
        return dx::bwor(dx::abs(magnitude), dx::bwand(dx::msb, sign));
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, A> magnitude, basic_vector<E, A> sign) noexcept {
        if constexpr (unqualified_canonical_copysign<basic_vector<E, A>>) {
            if consteval {
                return fallback(magnitude, sign);
            } else {
                return copysign(internal::abi<A>, magnitude, sign);
            }
        } else {
            return fallback(magnitude, sign);
        }
    }

    template <simd_abi LA, simd_abi RA, simd_element_for<LA> E>
    requires simd_element_for<E, RA> &&
        (!floating_point<E> || different_from<LA, RA>) &&
        unqualified_canonical_copysign<basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, LA> magnitude, basic_vector<E, RA> sign) noexcept {
        using A = common_abi_t<LA, RA>;
        return copysign(internal::abi<A>, magnitude, sign);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_copysign<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L magnitude, R sign) noexcept {
        if constexpr (unqualified_extended_copysign<L, R>) {
            return copysign(magnitude, sign);
        } else {
            return operator()(
                dx::to_canonical(magnitude), dx::to_canonical(sign));
        }
    }

    template <simd_vector R, broadcastable_to<R> L>
    requires requires(copysign_t op, R val) { op(val, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L magnitude, R sign) noexcept {
        if constexpr (canonical_vector<R>) {
            if constexpr (requires {
                              copysign(internal::abi<R>, magnitude, sign);
                          }) {
                if consteval {
                    return operator()(dx::broadcast<R>(magnitude), sign);
                } else {
                    return copysign(internal::abi<R>, magnitude, sign);
                }
            } else {
                return operator()(dx::broadcast<R>(magnitude), sign);
            }
        } else if constexpr (requires { copysign(magnitude, sign); }) {
            return copysign(magnitude, sign);
        } else {
            static_assert(
                decayable_vector_for<R, operation_category::lane_agnostic>);
            return operator()(magnitude, dx::to_canonical(sign));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::copysign_t copysign{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
