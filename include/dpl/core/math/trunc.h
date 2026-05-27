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
#  include "dpl/core/concepts/simd_vector.h"
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

struct trunc_t;

template <typename T>
concept unqualified_canonical_trunc = requires(T val) {
    {
        round(internal::abi<T>, val, rounding::to_zero)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_trunc = requires(T val) {
    {
        round(val, rounding::to_zero)
    } -> extended_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_trunc = unqualified_extended_trunc<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<trunc_t, canonical_type_t<T>>);

template <typename T>
concept unqualified_canonical_truncne = requires(T val) {
    {
        round(internal::abi<T>, val, rounding::to_zero | rounding::no_exc)
    } -> equivalent_simd_as<T>;
};

template <typename T>
concept unqualified_extended_truncne = requires(T val) {
    {
        round(val, rounding::to_zero | rounding::no_exc)
    } -> equivalent_simd_as<T>;
};

template <typename T>
concept unqualified_truncne = unqualified_extended_trunc<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<trunc_t, canonical_type_t<T>, rounding::no_exc_t>);

struct trunc_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
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
        auto const result = dx::select(m, dx::zero, val);
        return dx::select(exp >= width ||
                dx::bwand(dx::reinterpret<uint>(val), m) == dx::zero,
            val, result);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_trunc<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return round(internal::abi<A>, val, rounding::to_zero);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (scalable_abi<A> || !floating_point<E>) &&
        unqualified_canonical_trunc<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return round(internal::abi<A>, val, rounding::to_zero);
    }

    template <extended_vector T>
    requires unqualified_trunc<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_trunc<T>) {
            return round(val, rounding::to_zero);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        if constexpr (unqualified_canonical_truncne<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return round(internal::abi<A>, val,
                    rounding::to_zero | rounding::no_exc);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (scalable_abi<A> || !floating_point<E>) &&
        unqualified_canonical_truncne<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        return round(
            internal::abi<A>, val, rounding::to_zero | rounding::no_exc);
    }

    template <extended_vector T>
    requires unqualified_truncne<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, rounding::no_exc_t) noexcept {
        if constexpr (unqualified_extended_trunc<T>) {
            return round(val, rounding::to_zero | rounding::no_exc);
        } else {
            return operator()(dx::to_canonical(val), rounding::no_exc);
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
