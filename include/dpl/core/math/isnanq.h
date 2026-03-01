// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/compatible_mask_for.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/mantissa_width.h"
#  include "dpl/core/constants/value_bits.h"
#  include "dpl/core/operations/abs.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/std/bit/bit_cast.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void isnanq(...) noexcept = delete;

struct isnanq_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd_mask<E, A> DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        constexpr signed_rep_t<E> signaling_bit = signed_rep_t<E>(1)
            << (dx::mantissa_width_v<E> - 1);
        constexpr signed_rep_t<E> max_snan =
            __DPL bit_cast<signed_rep_t<E>>(dx::value_bits_v<E>) ^
            signaling_bit;

        using sint = signed_rep_t<E>;
        return dx::cmpgt(dx::reinterpret<sint>(dx::abs(val)), max_snan);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr to_simd_mask_type_t<T> operator()(T arg) noexcept {
        if constexpr (requires { isnanq(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return isnanq(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept
        -> compatible_mask_for<T> auto {
        if constexpr (requires { isnanq(internal::abi<T>, arg); }) {
            return isnanq(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::isnanq_t isnanq{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
