// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/arithmetic.h"
#include "dpl/core/operations/bit.h"
#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <auto>
void negate(...) noexcept = delete;

template <typename C, typename T, typename A = common_abi_t<T>>
concept unqualified_mnegate = requires(C mask, T val) {
    {
        negate(internal::abi<A>, mask, val)
    } -> simd_with<common_arithmetic_type_t<typename T::value_type,
                       typename T::value_type>,
        A>;
};

template <auto V, typename T>
concept unqualified_negatei = requires(T val) {
    {
        negate<V>(internal::abi<T>, val)
    } -> equivalent_simd_as<common_arithmetic_simd_t<T, T>>;
};

struct negate_t {
private:
    template <typename T>
    using negated_type DPL_NODEBUG = common_arithmetic_type_t<T, T>;
    template <typename T>
    using negated_simd DPL_NODEBUG = common_arithmetic_simd_t<T, T>;

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd<E, A> val) noexcept {
        using T = negated_type<E>;
        if constexpr (enumeration<E>) {
            return operator()(dx::reinterpret<T>(val));
        } else {
            return dx::reinterpret<T>(dx::subtract(dx::zero, val));
        }
    }

    template <typename C, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<C, A> mask, basic_simd<E, A> val) noexcept {
        using T = negated_type<E>;
        return dx::select(mask, fallback(val), dx::reinterpret<T>(val));
    }

    template <integral auto V, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_simd<E, A> val) noexcept {
        static constexpr immediate_mask<simd_abi_traits<E, A>::size, V> mask{};
        if constexpr (all_of(mask)) {
            return fallback(val);
        } else if constexpr (none_of(mask)) {
            using T = negated_type<E>;
            return dx::reinterpret<T>(val);
        } else {
            using T = negated_type<E>;
            return dx::selecti<V>(
                dx::reinterpret<T>(dx::subtract(dx::zero, val)),
                dx::reinterpret<T>(val));
        }
    }

public:
    template <arithmetic_simd T>
    requires fixed_width_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(T val) noexcept {
        if constexpr (unqualified_negate<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallback(val);
                } else {
                    return negate(internal::abi<T>, val);
                }
            } else {
                return negate(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }

    template <arithmetic_simd T>
    requires scalable_simd<T> &&
        (unqualified_negate<T> || unqualified_negate<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(T val) noexcept {
        if constexpr (unqualified_negate<T>) {
            return negate(internal::abi<T>, val);
        } else {
            return negate(internal::abi<T>, dx::to_basic_type(val));
        }
    }

    template <arithmetic_simd T, compatible_mask_with<T> M>
    requires same_abi_simd_as<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(M mask, T val) noexcept {
        using A = typename T::abi_type;
        if constexpr (unqualified_mnegate<M, T, A>) {
            if constexpr (basic_simd_type<T> && basic_simd_mask_type<M>) {
                if consteval {
                    return fallback(mask, val);
                } else {
                    return negate(internal::abi<A>, mask, val);
                }
            } else {
                return negate(internal::abi<A>, mask, val);
            }
        } else if constexpr (basic_simd_type<T> && basic_simd_mask_type<M>) {
            return fallback(mask, val);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(val));
        }
    }

    template <arithmetic_simd T, compatible_mask_with<T> M>
    requires (!same_abi_simd_as<T, M> &&
        (unqualified_mnegate<M, T> ||
            unqualified_mnegate<basic_type_t<M>, basic_type_t<T>>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(M mask, T val) noexcept {
        using A = common_abi_t<M, T>;
        if constexpr (unqualified_mnegate<M, T>) {
            return negate(internal::abi<A>, mask, val);
        } else {
            return negate(internal::abi<A>, dx::to_basic_type(mask),
                dx::to_basic_type(val));
        }
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(M mask, T val) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (unqualified_negate<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallbacki<V>(val);
                } else {
                    return negate<V>(internal::abi<T>, val);
                }
            } else {
                return negate<V>(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallbacki<V>(val);
        } else {
            return operator()(mask, dx::to_basic_type(val));
        }
    }
};

template <auto V>
struct negatei_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_immediate_mask_t<T, V>;

public:
    template <simd_class T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<negate_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return negate_t::operator()(mask, arg);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::negate_t negate{};
DPL_EXPORT template <auto V>
inline constexpr internal::negatei_t<V> negatei{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
