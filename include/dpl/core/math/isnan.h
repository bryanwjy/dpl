// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/compatible_mask_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/constants/infinity.h"
#  include "dpl/core/constants/value_bits.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/type_traits/representation.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void isnan(...) noexcept = delete;

struct isnan_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg) noexcept {
        using uint = unsigned_representation_t<E>;
        constexpr auto inf_bits =
            dx::reinterpret<uint>(dx::infinity_v<basic_vector<E, A>>);
        constexpr auto abs_bits =
            dx::reinterpret<uint>(dx::value_bits_v<basic_vector<E, A>>);
        auto const abs_val = dx::reinterpret<uint>(dx::bwand(arg, abs_bits));
        return dx::cmpgt(abs_val, inf_bits);
    }

public:
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr make_simd_mask_type_t<T> operator()(T arg) noexcept {
        if constexpr (requires {
                          {
                              isnan(internal::abi<T>, arg)
                          } -> compatible_mask_with<T>;
                      }) {
            if constexpr (canonical_vector<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return isnan(internal::abi<T>, arg);
                }
            } else {
                return isnan(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_vector<T>) {
            return fallback(arg);
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
