// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/compatible_mask_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/infinity.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void isnormal(...) noexcept = delete;

struct isnormal_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd_mask<E, A> DPL_VECTORCALL
        fallback(basic_simd<E, A> arg) noexcept {
        auto const val = dx::bwand(arg, dx::infinity);
        return val != dx::infinity && val > dx::zero;
    }

public:
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr make_simd_mask_type_t<T> operator()(T arg) noexcept {
        if constexpr (requires {
                          {
                              isnormal(internal::abi<T>, arg)
                          } -> compatible_mask_with<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return isnormal(internal::abi<T>, arg);
                }
            } else {
                return isnormal(internal::abi<T>, arg);
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
DPL_EXPORT inline constexpr internal::isnormal_t isnormal{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
