// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/minmax.h"
#include "dpl/core/operations/negate.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void abs(...) noexcept = delete;
template <auto>
void abs(...) noexcept = delete;

template <typename T>
concept unqualified_abs = requires(T val) {
    {
        abs(internal::abi<T>, val)
    } -> equivalent_simd_as<common_arithmetic_simd_t<T, T>>;
};

template <typename C, typename T, typename A = common_abi_t<T>>
concept unqualified_mabs = requires(C mask, T val) {
    {
        abs(internal::abi<A>, mask, val)
    } -> simd_with<common_arithmetic_type_t<typename T::value_type,
                       typename T::value_type>,
        A>;
};

template <auto V, typename T>
concept unqualified_absi = requires(T val) {
    {
        abs<V>(internal::abi<T>, val)
    } -> equivalent_simd_as<common_arithmetic_simd_t<T, T>>;
};

struct abs_t {
private:
    template <typename T>
    using negated_type DPL_NODEBUG = common_arithmetic_type_t<T, T>;
    template <typename T>
    using negated_simd DPL_NODEBUG = common_arithmetic_simd_t<T, T>;

    template <arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        using T = negated_type<E>;
        if constexpr (enumeration<E>) {
            return operator()(dx::reinterpret<underlying_type_t<E>>(val));
        } else if constexpr (unsigned_integral_simd<E>) {
            return val;
        } else {
            return dx::max(dx::reinterpret<T>(val), dx::negate(val));
        }
    }

    template <simd_element C, arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd_mask<E, A> mask, basic_simd<E, A> val) noexcept {
        using T = negated_type<E>;
        return dx::max(dx::reinterpret<T>(val), dx::negate(mask, val));
    }

    template <integral auto V, arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<E, A> DPL_VECTORCALL
        fallbacki(basic_simd<E, A> val) noexcept {
        static constexpr immediate_mask<element_count<E, A>, V> mask{};
        if constexpr (all_of(mask)) {
            return fallback(val);
        } else if constexpr (none_of(mask)) {
            using T = negated_type<E>;
            return dx::reinterpret<T>(val);
        } else {
            using T = negated_type<E>;
            return dx::max(dx::reinterpret<T>(val), dx::negatei<V>(val));
        }
    }

public:
    template <arithmetic_simd T>
    requires fixed_width_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(T val) noexcept {
        if constexpr (unqualified_abs<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallback(val);
                } else {
                    return abs(internal::abi<T>, val);
                }
            } else {
                return abs(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }

    template <arithmetic_simd T>
    requires scalable_simd<T> &&
        (unqualified_abs<T> || unqualified_abs<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(T val) noexcept {
        if constexpr (unqualified_abs<T>) {
            return abs(internal::abi<T>, val);
        } else {
            return abs(internal::abi<T>, dx::to_basic_type(val));
        }
    }

    template <arithmetic_simd T, compatible_mask_with<T> M>
    requires same_abi_simd_as<T, M> && fixed_width_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(M mask, T val) noexcept {
        using A = typename T::abi_type;
        if constexpr (unqualified_mabs<M, T, A>) {
            if constexpr (basic_simd_type<T> && basic_simd_mask_type<M>) {
                if consteval {
                    return fallback(mask, val);
                } else {
                    return abs(internal::abi<A>, mask, val);
                }
            } else {
                return abs(internal::abi<A>, mask, val);
            }
        } else if constexpr (basic_simd_type<T> && basic_simd_mask_type<M>) {
            return fallback(mask, val);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(val));
        }
    }

    template <arithmetic_simd T, compatible_mask_with<T> M>
    requires (!same_abi_simd_as<T, M> || scalable_simd<T>) &&
        (unqualified_mabs<M, T> ||
            unqualified_mabs<basic_type_t<M>, basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(M mask, T val) noexcept {
        using A = common_abi_t<M, T>;
        if constexpr (unqualified_mabs<M, T>) {
            return abs(internal::abi<A>, mask, val);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(val));
        }
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(M mask, T val) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (unqualified_abs<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallbacki<V>(val);
                } else {
                    return abs<V>(internal::abi<T>, val);
                }
            } else {
                return abs<V>(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallbacki<V>(val);
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
