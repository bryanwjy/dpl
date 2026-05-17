// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/compliance.h"
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/ilogb.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/utility/rounding.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace mx = datapar::fmath;

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
        using sint = signed_representation_t<E>;
        using uint = unsigned_representation_t<E>;
        static constexpr auto width =
            dx::broadcast<sint, A>(sizeof(E) * char_bit_v);
        static constexpr auto margin = width - dx::mantissa_width_v<E>;
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
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (unqualified_trunc<T>) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return round(internal::abi<T>, val, rounding::to_zero);
                } else {
                    return fallback(val);
                }
            } else {
                return round(internal::abi<T>, val, rounding::to_zero);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T val, rounding::no_exc_t tag [[maybe_unused]]) noexcept {
        if constexpr (unqualified_trunc_noexc<T>) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return round(internal::abi<T>, val,
                        rounding::to_zero | rounding::no_exc);
                } else {
                    return fallback(val);
                }
            } else {
                return round(internal::abi<T>, val,
                    rounding::to_zero | rounding::no_exc);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val), tag);
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::trunc_t trunc{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
