// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bit.h"
#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void abs(...) noexcept = delete;
template <auto>
void abs(...) noexcept = delete;

struct abs_t {
private:
    template <typename T>
    using negated_type DPL_NODEBUG = common_arithmetic_type_t<T, T>;
    template <typename T>
    using negated_simd DPL_NODEBUG = common_arithmetic_simd_t<T, T>;

    template <arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        if constexpr (floating_point<E>) {
            return dx::reinterpret<E>(dx::bwandnot(val, dx::msb));
        } else {
            using T = negated_type<E>;
            return internal::transform<basic_simd<T, A>>(val,
                [](auto val) { return static_cast<T>(val < 0 ? -val : val); });
        }
    }

    template <integral auto V, arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<E, A> DPL_VECTORCALL
        fallbacki(basic_simd<E, A> val) noexcept {
        static constexpr immediate_mask<element_count<E, A>, V> mask{};
        if constexpr (all_of(mask)) {
            return fallback(val);
        } else if constexpr (none_of(mask)) {
            using T = negated_type<E>;
            return dx::reinterpret<T>(val);
        } else if constexpr (floating_point<E>) {
            constexpr auto nmask = ~mask;
            return dx::reinterpret<E>(dx::bwandnot(
                val, dx::bit_clear(nmask, dx::msb_v<basic_simd<E, A>>)));
        } else {
            using T = negated_type<E>;
            return internal::itransform<basic_simd<T, A>>(
                val, [](size_t idx, auto val) {
                    return static_cast<T>(mask[idx] && val < 0 ? -val : val);
                });
        }
    }

public:
    template <basic_simd_type T>
    requires arithmetic_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr negated_simd<T> DPL_VECTORCALL operator()(T val) noexcept {
        if constexpr (unsigned_integral_simd<T>) {
            return val;
        } else if constexpr (requires { abs(internal::abi<T>, val); }) {
            if consteval {
                return fallback(val);
            } else {
                return abs(internal::abi<T>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <arithmetic_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T val) noexcept
        -> equivalent_simd_as<negated_simd<T>> auto {
        if constexpr (requires { abs(internal::abi<T>, val); }) {
            return abs(internal::abi<T>, val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }

    template <basic_simd_mask_type M, basic_simd_type T>
    requires arithmetic_simd<T> && compatible_mask_for<M, T> &&
        same_abi_simd_as<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr negated_simd<T> DPL_VECTORCALL operator()(
        M mask, T val) noexcept {
        if constexpr (unsigned_integral_simd<T>) {
            return val;
        } else if constexpr (requires { abs(internal::abi<T>, mask, val); }) {
            if consteval {
                return dx::select(
                    mask, fallback(val), dx::reinterpret<negated_type<T>>(val));
            } else {
                return abs(internal::abi<T>, mask, val);
            }
        } else {
            return dx::select(
                mask, fallback(val), dx::reinterpret<negated_type<T>>(val));
        }
    }

    template <basic_simd_mask_type M, basic_simd_type T>
    requires arithmetic_simd<T> && compatible_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T val) noexcept
        -> equivalent_simd_as<negated_simd<T>> auto {
        using A = common_abi_t<M, T>;
        return abs(internal::abi<A>, mask, val);
    }

    template <simd_type T, compatible_mask_for<T> M>
    requires arithmetic_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T val) noexcept
        -> equivalent_simd_as<negated_simd<T>> auto {
        if constexpr (requires { abs(internal::abi<T>, mask, val); }) {
            return abs(internal::abi<T>, mask, val);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(val));
        }
    }

    template <basic_simd_type T, immediate_mask_for<T> M>
    requires arithmetic_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr negated_simd<T> DPL_VECTORCALL operator()(
        M mask, T val) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (unsigned_integral_simd<T>) {
            return val;
        } else if constexpr (requires { abs<V>(internal::abi<T>, val); }) {
            if consteval {
                return fallbacki<V>(val);
            } else {
                return abs<V>(internal::abi<T>, val);
            }
        } else {
            return fallbacki<V>(val);
        }
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T val) noexcept
        -> equivalent_simd_as<negated_simd<T>> auto {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { abs<V>(internal::abi<T>, val); }) {
            return abs<V>(internal::abi<T>, val);
        } else {
            return operator()(mask, dx::to_basic_type(val));
        }
    }
};

template <auto V>
struct absi_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <simd_class T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<abs_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return abs_t::operator()(mask, arg);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::abs_t abs{};
DPL_EXPORT template <auto V>
inline constexpr internal::absi_t<V> absi{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
