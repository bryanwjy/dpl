// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/core/operations/abs.h"
#  include "dpl/core/operations/bitwise.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void copysign(...) noexcept = delete;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_copysign = requires(L lhs, R rhs) {
    { copysign(internal::abi<A>, lhs, rhs) } -> floating_point_simd_with_abi<A>;
};

struct copysign_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<E, A> DPL_VECTORCALL
        fallback(basic_simd<E, A> magnitude, basic_simd<E, A> sign) noexcept {
        return dx::bwor(dx::abs(magnitude), dx::bwand(dx::msb, sign));
    }

public:
    template <floating_point_simd L, common_float_simd_with<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L magnitude, R sign) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_copysign<L, R, A>) {
            if constexpr (basic_simd_type<L>) {
                if not consteval {
                    return copysign(internal::abi<A>, magnitude, sign);
                } else {
                    return fallback(magnitude, sign);
                }
            } else {
                return copysign(internal::abi<A>, magnitude, sign);
            }
        } else if constexpr (basic_simd_type<L>) {
            return fallback(magnitude, sign);
        } else {
            return operator()(
                dx::to_basic_type(magnitude), dx::to_basic_type(sign));
        }
    }

    template <floating_point_simd R, broadcastable_to<R> L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L magnitude, R sign) noexcept {
        if constexpr (unqualified_copysign<L, R, typename R::abi_type>) {
            if constexpr (basic_simd_type<R>) {
                if consteval {
                    return operator()(dx::broadcast<R>(magnitude), sign);
                } else {
                    return copysign(internal::abi<R>, magnitude, sign);
                }
            } else {
                return copysign(internal::abi<R>, magnitude, sign);
            }
        } else if constexpr (basic_simd_type<R>) {
            return operator()(dx::broadcast<R>(magnitude), sign);
        } else {
            return operator()(magnitude, dx::to_basic_type(sign));
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
