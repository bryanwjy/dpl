// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fixup.h"
#include "dpl/core/math/frexp.h"
#include "dpl/core/math/internal/accuracy.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/rsqrt2.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace mx = datapar::fmath;

void rsqrt(...) noexcept = delete;

struct rsqrt_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        constexpr auto inv_sqrt2 = dx::broadcast<E, A>(mx::rsqrt2(dx::one));

        auto const decomp = dx::frexp(val);
        auto const remtwo = decomp.exp & dx::one;
        auto const reduced = mx::rsqrt2(mx::accuracy::speed, decomp.fr);
        auto result = mx::ldexp(mx::compliance::unsafe, reduced,
            -((decomp.exp - dx::one) >> imm<1>));
        result *= dx::select(remtwo == dx::zero, inv_sqrt2, dx::one);

        return dx::fixup(val, result,
            fpfix::condition<fpfix::nan, fpfix::revert>       //
                | fpfix::condition<fpfix::infinity, dx::zero> //
                | fpfix::condition<fpfix::negative, dx::nan>);
    }

public:
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              rsqrt(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return rsqrt(internal::abi<T>, val);
                } else {
                    return fallback(val);
                }
            } else {
                return rsqrt(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::rsqrt_t rsqrt{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
