// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/compatible_mask_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/mantissa_width.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/std/bit/char_bit.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void isnans(...) noexcept = delete;

struct isnans_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd_mask<E, A> DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        using uint = unsigned_rep_t<E>;
        using sint = signed_rep_t<E>;
        constexpr auto shift = sizeof(E) * char_bit_v - dx::mantissa_width_v<E>;
        return dx::cmpgt(
            dx::bwshift_left(dx::reinterpret<sint>(val), imm<shift>), dx::zero);
    }

public:
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr make_simd_mask_type_t<T> operator()(T arg) noexcept {
        if constexpr (requires {
                          {
                              isnans(internal::abi<T>, arg)
                          } -> compatible_mask_with<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return isnans(internal::abi<T>, arg);
                }
            } else {
                return isnans(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::isnans_t isnans{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
