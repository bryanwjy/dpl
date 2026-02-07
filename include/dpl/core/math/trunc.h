// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/compliance.h"
#include "dpl/core/math/internal/ilogb.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/to_integral.h"
#  include "dpl/core/utility/rounding.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace mx = datapar::fmath;

void trunc(...) noexcept = delete;

template <typename T>
concept unqualified_trunc =
    requires(T val) { round(internal::abi<T>, val, rounding::to_zero); };

template <typename T>
concept unqualified_trunc_noexc = requires(T val) {
    round(internal::abi<T>, val, rounding::to_zero | rounding::no_exc);
};

struct trunc_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        // Based on musl libm
        using sint = to_signed_integral_t<E>;
        using uint = to_unsigned_integral_t<E>;
        static constexpr auto width =
            dx::broadcast<sint, A>(sizeof(E) * char_bit_v);
        static constexpr auto margin = width - dx::digits<E>;
        auto const exp = [](auto exp) {
            return dx::select(exp < margin, dx::one, exp);
        }(mx::ilogb(mx::compliance::unsafe, val) + margin);

        auto const m = dx::all_bits_v<decltype(val)> >> exp;
        auto const result = dx::bit_drop(m, val);
        return dx::select(exp >= width ||
                dx::bwand(dx::reinterpret<uint>(val), m) == dx::zero,
            val, result);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (unqualified_trunc<T>) {
            if not consteval {
                return round(internal::abi<T>, val, rounding::to_zero);
            } else {
                return fallback(val);
            }
        } else {
            return fallback(val);
        }
    }

    template <floating_point_simd T>
    requires (!basic_simd_type<T>) && unqualified_trunc<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept
        -> equivalent_simd_as<T> auto {
        return round(internal::abi<T>, val, rounding::to_zero);
    }

    template <floating_point_simd T>
    requires (!basic_simd_type<T> && !unqualified_trunc<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        return operator()(dx::to_basic_type(val));
    }

    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, rounding::no_exc_t) noexcept {
        if constexpr (unqualified_trunc_noexc<T>) {
            if not consteval {
                return round(internal::abi<T>, val,
                    rounding::to_pos_inf | rounding::no_exc);
            } else {
                return fallback(val);
            }
        } else {
            return fallback(val);
        }
    }

    template <floating_point_simd T>
    requires (!basic_simd_type<T>) && unqualified_trunc_noexc<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, rounding::no_exc_t) noexcept
        -> equivalent_simd_as<T> auto {
        return round(
            internal::abi<T>, val, rounding::to_pos_inf | rounding::no_exc);
    }

    template <floating_point_simd T>
    requires (!basic_simd_type<T> && !unqualified_trunc_noexc<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, rounding::no_exc_t tag) noexcept {
        return operator()(dx::to_basic_type(val), tag);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::trunc_t trunc{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
