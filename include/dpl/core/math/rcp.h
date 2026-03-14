// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fixup.h"
#include "dpl/core/math/fma.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/utility/fpfix.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void rcp(...) noexcept = delete;

struct rcp_t {
private:
    template <floating_point E>
    static constexpr auto useed = []() {
        if constexpr (common_float_with<E, double>) {
            return unsigned_rep_t<E>(0x7FDE6238DA3C2118);
        } else if constexpr (common_float_with<E, float>) {
            return unsigned_rep_t<E>(0x7EF311C3);
        } else if constexpr (brain_float<E>) {
            return unsigned_rep_t<E>(0x7EF3);
        } else {
            static_assert(sizeof(E) == 2);
            return unsigned_rep_t<E>(0x7800);
        }
    }();

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto refine(
        basic_simd<E, A> y, basic_simd<E, A> x) noexcept {
        return y * dx::fnmadd(x, y, 2.0);
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_simd<float, A> approximate(
        basic_simd<float, A> val) noexcept {
        auto const seed = useed<E> - dx::reinterpret<unsigned_rep_t<E>>(val);
        return refine(dx::reinterpret<E>(seed), val);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<float, A> val) noexcept {
        auto const result = approximate(val);
        // nan is implicitly handled
        return dx::fixup(val, result,
            fpfix::condition<fpfix::infinity, dx::zero> |
                fpfix::condition<fpfix::zero, fpfix::signed_inf>);
    }

public:
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              rcp(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return rcp(internal::abi<T>, val);
                } else {
                    return fallback(val);
                }
            } else {
                return rcp(internal::abi<T>, val);
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
DPL_EXPORT inline constexpr internal::rcp_t rcp{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
