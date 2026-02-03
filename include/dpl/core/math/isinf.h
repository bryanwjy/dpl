// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/compatible_mask_for.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/infinity.h"
#  include "dpl/core/constants/value_bits.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void isinf(...) noexcept = delete;

struct isinf_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask<E, A> fallback(basic_simd<E, A> arg) noexcept {
        return dx::cmpeq(dx::bwand(arg, dx::value_bits), dx::infinity);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr to_simd_mask_type_t<T> DPL_VECTORCALL operator()(
        T arg) noexcept {
        if constexpr (requires { isinf(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return isinf(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept
        -> compatible_mask_for<T> auto {
        if constexpr (requires { isinf(internal::abi<T>, arg); }) {
            return isinf(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
DPL_EXPORT inline constexpr internal::isinf_t isinf{};
}
DPL_DEFAULT_NAMESPACE_END
