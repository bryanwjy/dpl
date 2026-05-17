// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/common_bits_with.h"
#  include "dpl/core/concepts/integral_simd.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/constants/all_bits.h"
#  include "dpl/core/type_traits/common_bits_type.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/type_traits/enable_if.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwor(...) noexcept = delete;
void bwand(...) noexcept = delete;
void bwxor(...) noexcept = delete;
void bwandnot(...) noexcept = delete;
void bwornot(...) noexcept = delete;
void bwnot(...) noexcept = delete;
void bwshift_left(...) noexcept = delete;
void bwshift_right(...) noexcept = delete;
template <size_t>
void bwshift_left(...) noexcept = delete;
template <size_t>
void bwshift_right(...) noexcept = delete;

template <typename T, typename L, typename R>
concept bitwise_result = common_bits_simd_with<T, common_bits_simd_t<L, R>> &&
    same_abi_as<common_abi_t<L, R>, typename T::abi_type>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bwor = requires(L lhs, R rhs) {
    { bwor(internal::abi<A>, lhs, rhs) } -> bitwise_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bwand = requires(L lhs, R rhs) {
    { bwand(internal::abi<A>, lhs, rhs) } -> bitwise_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bwxor = requires(L lhs, R rhs) {
    { bwxor(internal::abi<A>, lhs, rhs) } -> bitwise_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bwandnot = requires(L lhs, R rhs) {
    { bwandnot(internal::abi<A>, lhs, rhs) } -> bitwise_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bwornot = requires(L lhs, R rhs) {
    { bwornot(internal::abi<A>, lhs, rhs) } -> bitwise_result<L, R>;
};

template <typename T, typename L, typename R>
concept mbitwise_result = common_size_simd_with<T, common_size_simd_t<L, R>> &&
    same_abi_as<common_abi_t<L, R>, typename T::abi_type>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mbwor = requires(L lhs, R rhs) {
    { bwor(internal::abi<A>, lhs, rhs) } -> mbitwise_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mbwand = requires(L lhs, R rhs) {
    { bwand(internal::abi<A>, lhs, rhs) } -> mbitwise_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mbwxor = requires(L lhs, R rhs) {
    { bwxor(internal::abi<A>, lhs, rhs) } -> mbitwise_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mbwandnot = requires(L lhs, R rhs) {
    { bwandnot(internal::abi<A>, lhs, rhs) } -> mbitwise_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mbwornot = requires(L lhs, R rhs) {
    { bwornot(internal::abi<A>, lhs, rhs) } -> mbitwise_result<L, R>;
};

template <typename T>
concept unqualified_bwnot = requires(T val) {
    { bwnot(internal::abi<T>, val) } -> equivalent_simd_as<T>;
};

template <typename T>
concept unqualified_mbwnot = requires(T val) {
    { bwnot(internal::abi<T>, val) } -> equivalent_mask_as<T>;
};

template <typename T, typename L, typename R>
concept vbitshift_result =
    simd_with<T, typename L::value_type, common_abi_t<L, R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_vbwshift_left = requires(L lhs, R rhs) {
    { bwshift_left(internal::abi<A>, lhs, rhs) } -> vbitshift_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_vbwshift_right = requires(L lhs, R rhs) {
    { bwshift_right(internal::abi<A>, lhs, rhs) } -> vbitshift_result<L, R>;
};

template <typename L>
concept unqualified_bwshift_left = requires(L lhs, int rhs) {
    { bwshift_left(internal::abi<L>, lhs, rhs) } -> equivalent_class_as<L>;
};

template <typename L>
concept unqualified_bwshift_right = requires(L lhs, int rhs) {
    { bwshift_right(internal::abi<L>, lhs, rhs) } -> equivalent_class_as<L>;
};

template <typename L, typename R>
concept unqualified_bwshift_lefti = requires(L lhs) {
    { bwshift_left<R::value>(internal::abi<L>, lhs) } -> equivalent_class_as<L>;
};

template <typename L, typename R>
concept unqualified_bwshift_righti = requires(L lhs) {
    {
        bwshift_right<R::value>(internal::abi<L>, lhs)
    } -> equivalent_class_as<L>;
};

template <simd_type L, simd_type R>
using bwsimd_for DPL_NODEBUG =
    common_bits_simd_t<basic_type_t<L>, basic_type_t<R>>;
template <simd_mask_type L, simd_mask_type R>
using bwmask_for DPL_NODEBUG =
    common_size_simd_t<basic_type_t<L>, basic_type_t<R>>;

struct bwor_t : binary_operation_base<bwor_t> {
private:
    friend binary_operation_base<bwor_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept
    requires requires {
        { bwor(internal::abi<A>, left, right) } -> simd_class_with_abi<A>;
    }
    {
        return bwor(internal::abi<A>, left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
                auto const promoted = __DPL bit_cast<bit_type>(lhs) |
                    __DPL bit_cast<bit_type>(rhs);
                return __DPL bit_cast<T>(static_cast<bit_type>(promoted));
            },
            lhs, rhs);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<L, A> left, basic_simd_mask<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            [](bool lhs, bool rhs) { return lhs || rhs; }, left, right);
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bwor<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(bwor(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return bwor(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bwor(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_bits_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_bwor<L, R> ||
            unqualified_bwor<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_bwor<L, R>) {
            return bwor(internal::abi<A>, lhs, rhs);
        } else {
            return bwor(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, common_size_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_mbwor<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_mask_type<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(bwor(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return bwor(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bwor(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_mask_type<L> &&
            basic_simd_mask_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, common_size_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mbwor<L, R> ||
            unqualified_mbwor<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mbwor<L, R>) {
            return bwor(internal::abi<A>, lhs, rhs);
        } else {
            return bwor(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<bwor_t>::operator();
};

struct bwand_t : binary_operation_base<bwand_t> {
private:
    friend binary_operation_base<bwand_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept
    requires requires {
        { bwand(internal::abi<A>, left, right) } -> simd_class_with_abi<A>;
    }
    {
        return bwand(internal::abi<A>, left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
                auto const promoted = __DPL bit_cast<bit_type>(lhs) &
                    __DPL bit_cast<bit_type>(rhs);
                return __DPL bit_cast<T>(static_cast<bit_type>(promoted));
            },
            left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<L, A> left, basic_simd_mask<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            [](bool lhs, bool rhs) { return lhs && rhs; }, left, right);
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bwand<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(bwand(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return bwand(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bwand(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_bits_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_bwand<L, R> ||
            unqualified_bwand<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_bwand<L, R>) {
            return bwand(internal::abi<A>, lhs, rhs);
        } else {
            return bwand(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, common_size_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_mbwand<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_mask_type<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(bwand(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return bwand(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bwand(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_mask_type<L> &&
            basic_simd_mask_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, common_size_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mbwand<L, R> ||
            unqualified_mbwand<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mbwand<L, R>) {
            return bwand(internal::abi<A>, lhs, rhs);
        } else {
            return bwand(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<bwand_t>::operator();
};

struct bwxor_t : binary_operation_base<bwxor_t> {
private:
    friend binary_operation_base<bwxor_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept
    requires requires {
        { bwxor(internal::abi<A>, left, right) } -> simd_class_with_abi<A>;
    }
    {
        return bwxor(internal::abi<A>, left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
                auto const promoted = __DPL bit_cast<bit_type>(lhs) ^
                    __DPL bit_cast<bit_type>(rhs);
                return __DPL bit_cast<T>(static_cast<bit_type>(promoted));
            },
            left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<L, A> left, basic_simd_mask<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            [](bool lhs, bool rhs) { return lhs != rhs; }, left, right);
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bwxor<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(bwxor(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return bwxor(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bwxor(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_bits_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_bwxor<L, R> ||
            unqualified_bwxor<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_bwxor<L, R>) {
            return bwxor(internal::abi<A>, lhs, rhs);
        } else {
            return bwxor(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, common_size_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_mbwxor<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_mask_type<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(bwxor(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return bwxor(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bwxor(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_mask_type<L> &&
            basic_simd_mask_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, common_size_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mbwxor<L, R> ||
            unqualified_mbwxor<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mbwxor<L, R>) {
            return bwxor(internal::abi<A>, lhs, rhs);
        } else {
            return bwxor(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<bwxor_t>::operator();
};

struct bwandnot_t : binary_operation_base<bwandnot_t> {
private:
    friend binary_operation_base<bwandnot_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept
    requires requires {
        { bwandnot(internal::abi<A>, left, right) } -> simd_class_with_abi<A>;
    }
    {
        return bwandnot(internal::abi<A>, left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
                auto const promoted = __DPL bit_cast<bit_type>(lhs) &
                    ~__DPL bit_cast<bit_type>(rhs);
                return __DPL bit_cast<T>(static_cast<bit_type>(promoted));
            },
            left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<L, A> left, basic_simd_mask<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            [](bool lhs, bool rhs) { return lhs && !rhs; }, left, right);
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bwandnot<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(bwandnot(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return bwandnot(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bwandnot(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_bits_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_bwandnot<L, R> ||
            unqualified_bwandnot<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_bwandnot<L, R>) {
            return bwandnot(internal::abi<A>, lhs, rhs);
        } else {
            return bwandnot(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, common_size_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_mbwandnot<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_mask_type<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(bwandnot(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return bwandnot(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bwandnot(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_mask_type<L> &&
            basic_simd_mask_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, common_size_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mbwandnot<L, R> ||
            unqualified_mbwandnot<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mbwandnot<L, R>) {
            return bwandnot(internal::abi<A>, lhs, rhs);
        } else {
            return bwandnot(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<bwandnot_t>::operator();
};

struct bwornot_t : binary_operation_base<bwornot_t> {
private:
    friend binary_operation_base<bwornot_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept
    requires requires {
        { bwornot(internal::abi<A>, left, right) } -> simd_class_with_abi<A>;
    }
    {
        return bwornot(internal::abi<A>, left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
                auto const promoted = __DPL bit_cast<bit_type>(lhs) |
                    ~__DPL bit_cast<bit_type>(rhs);
                return __DPL bit_cast<T>(static_cast<bit_type>(promoted));
            },
            left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<L, A> left, basic_simd_mask<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            [](bool lhs, bool rhs) { return lhs || !rhs; }, left, right);
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bwornot<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(bwandnot(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return bwornot(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bwornot(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_bits_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_bwornot<L, R> ||
            unqualified_bwornot<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_bwornot<L, R>) {
            return bwornot(internal::abi<A>, lhs, rhs);
        } else {
            return bwornot(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, common_size_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_mbwornot<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_mask_type<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(bwornot(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return bwornot(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bwornot(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_mask_type<L> &&
            basic_simd_mask_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, common_size_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mbwornot<L, R> ||
            unqualified_mbwornot<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mbwornot<L, R>) {
            return bwornot(internal::abi<A>, lhs, rhs);
        } else {
            return bwornot(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<bwornot_t>::operator();
};

struct bwnot_t {
private:
    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> arg) noexcept {
        if constexpr (is_invocable_v<bwandnot_t, basic_simd<E, A>>) {
            constexpr auto all =
                dx::broadcast<basic_simd<E, A>>(dx::all_bits_v<E>);
            return bwandnot_t::operator()(dx::reinterpret<E>(arg), all);
        } else {
            using bit_type = bit_type_t<sizeof(E) * char_bit_v>;
            return internal::transform<basic_simd<E, A>>(
                [](auto arg) {
                    auto const promoted = ~__DPL bit_cast<bit_type>(arg);
                    return __DPL bit_cast<E>(static_cast<bit_type>(promoted));
                },
                arg);
        }
    }

    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd_mask<E, A> arg) noexcept {
        if constexpr (is_invocable_v<bwandnot_t, basic_simd_mask<E, A>>) {
            constexpr auto all = dx::broadcast<basic_simd_mask<E, A>>(true);
            return bwandnot_t::operator()(dx::reinterpret<E>(arg), all);
        } else {
            using bit_type = bit_type_t<sizeof(E) * char_bit_v>;
            return internal::transform<basic_simd_mask<E, A>>(
                [](auto arg) {
                    return __DPL bit_cast<E>(~__DPL bit_cast<bit_type>(arg));
                },
                arg);
        }
    }

public:
    template <fixed_width_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_bwnot<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    using E = typename decltype(bwnot(
                        internal::abi<T>, val))::value_type;
                    return dx::reinterpret<E>(fallback(val));
                } else {
                    return bwnot(internal::abi<T>, val);
                }
            } else {
                return bwnot(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }

    template <scalable_simd T>
    requires unqualified_bwnot<T> || unqualified_bwnot<basic_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_bwnot<T>) {
            return bwnot(internal::abi<T>, val);
        } else {
            return bwnot(internal::abi<T>, dx::to_basic_type(val));
        }
    }

    template <fixed_width_mask T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_mbwnot<T>) {
            if constexpr (basic_simd_mask_type<T>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(bwnot(
                        internal::abi<T>, val))>;
                    return dx::reinterpret<E>(fallback(val));
                } else {
                    return bwnot(internal::abi<T>, val);
                }
            } else {
                return bwnot(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_mask_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }

    template <scalable_simd T>
    requires unqualified_mbwnot<T> || unqualified_mbwnot<basic_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_mbwnot<T>) {
            return bwnot(internal::abi<T>, val);
        } else {
            return bwnot(internal::abi<T>, dx::to_basic_type(val));
        }
    }
};

template <typename E>
concept shiftable_type = floating_point<E> || integral<E> || enumeration<E>;
template <typename T>
concept shiftable_simd = simd_type<T> && shiftable_type<typename T::value_type>;
template <typename L, typename R>
concept vshiftable_simd_by =
    shiftable_simd<L> && integral_simd<R> && common_size_simd_with<L, R>;

struct bwshift_left_t {
private:
    template <shiftable_type L, integral R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> arg, basic_simd<R, A> shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        return internal::transform<basic_simd<L, A>>(
            [](auto lhs, auto rhs) {
                if constexpr (integral<L> && sizeof(L) < sizeof(int)) {
                    return static_cast<L>(__DPL bit_cast<bit_type>(lhs) << rhs);
                } else {
                    return __DPL bit_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) << rhs);
                }
            },
            arg, shift);
    }

    template <shiftable_type L, simd_abi A, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> arg, R shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        return internal::transform<basic_simd<L, A>>(
            [](auto lhs) {
                if constexpr (integral<L> && sizeof(L) < sizeof(int)) {
                    return static_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) << R::value);
                } else {
                    return __DPL bit_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) << R::value);
                }
            },
            arg);
    }

    template <shiftable_type L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> arg, size_t shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        return internal::transform<basic_simd<L, A>>(
            [shift](auto lhs) {
                if constexpr (integral<L> && sizeof(L) < sizeof(int)) {
                    return static_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) << shift);
                } else {
                    return __DPL bit_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) << shift);
                }
            },
            arg);
    }

    template <simd_element L, simd_abi A, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd_mask<L, A> arg, R shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        using mask_type = basic_simd_mask<L, A>;
        return []<size_t... Is>(mask_type arg, index_sequence<Is...>) {
            return dx::initialize<mask_type>(
                (Is >= R::value ? arg[Is - R::value] : false)...);
        }(arg, iota_sequence<L, A>);
    }

    template <simd_element L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd_mask<L, A> arg, size_t shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        using mask_type = basic_simd_mask<L, A>;
        return []<size_t... Is>(
                   mask_type arg, unsigned shift, index_sequence<Is...>) {
            return dx::initialize<mask_type>(
                (Is >= shift ? arg[Is - shift] : false)...);
        }(arg, static_cast<unsigned>(shift), iota_sequence<L, A>);
    }

public:
    template <integral_simd R, vshiftable_simd_by<R> L>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R shift) noexcept {
        using A = L::abi_type;
        if constexpr (unqualified_vbwshift_left<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(arg, shift);
                } else {
                    return bwshift_left(internal::abi<A>, arg, shift);
                }
            } else {
                return bwshift_left(internal::abi<A>, arg, shift);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(arg, shift);
        } else {
            return operator()(dx::to_basic_type(arg), dx::to_basic_type(shift));
        }
    }

    template <integral_simd R, vshiftable_simd_by<R> L>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_vbwshift_left<L, R> ||
            unqualified_vbwshift_left<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R shift) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_vbwshift_left<L, R>) {
            return bwshift_left(internal::abi<A>, arg, shift);
        } else {
            return bwshift_left(internal::abi<A>, dx::to_basic_type(arg),
                dx::to_basic_type(shift));
        }
    }

    template <fixed_width_class L, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L arg, R shift) noexcept {
        static_assert(R::value > 0 &&
            R::value < sizeof(typename L::value_type) * char_bit_v);
        if constexpr (unqualified_bwshift_lefti<L, R>) {
            if constexpr (basic_simd_class<L>) {
                if consteval {
                    return fallback(arg, shift);
                } else {
                    return bwshift_left<R::value>(internal::abi<L>, arg);
                }
            } else {
                return bwshift_left<R::value>(internal::abi<L>, arg);
            }
        } else if constexpr (basic_simd_class<L>) {
            return fallback(arg, shift);
        } else {
            return operator()(dx::to_basic_type(arg), shift);
        }
    }

    template <scalable_class L, integral_constant_like R>
    requires unqualified_bwshift_lefti<L, R> ||
        unqualified_bwshift_lefti<basic_type_t<L>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R shift) noexcept {
        static_assert(R::value > 0 &&
            R::value < sizeof(typename L::value_type) * char_bit_v);
        if constexpr (unqualified_bwshift_lefti<L, R>) {
            return bwshift_left<R::value>(internal::abi<L>, arg);
        } else {
            return bwshift_left<R::value>(
                internal::abi<L>, dx::to_basic_type(arg));
        }
    }

    template <fixed_width_class L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t shift) noexcept {
        if constexpr (unqualified_bwshift_left<L>) {
            if constexpr (basic_simd_class<L>) {
                if consteval {
                    return fallback(arg, shift);
                } else {
                    return bwshift_left(internal::abi<L>, arg, shift);
                }
            } else {
                return bwshift_left(internal::abi<L>, arg, shift);
            }
        } else if constexpr (basic_simd_class<L>) {
            return fallback(arg, shift);
        } else {
            return operator()(dx::to_basic_type(arg), shift);
        }
    }

    template <scalable_class L>
    requires unqualified_bwshift_left<L> ||
        unqualified_bwshift_left<basic_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t shift) noexcept {
        if constexpr (unqualified_bwshift_left<L>) {
            return bwshift_left(internal::abi<L>, arg, shift);
        } else {
            return bwshift_left(
                internal::abi<L>, dx::to_basic_type(arg), shift);
        }
    }
};

struct bwshift_right_t {
private:
    template <shiftable_type L, integral R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> arg, basic_simd<R, A> shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        return internal::transform<basic_simd<L, A>>(
            [](auto lhs, auto rhs) {
                if constexpr (integral<L> && sizeof(L) < sizeof(int)) {
                    return static_cast<L>(__DPL bit_cast<bit_type>(lhs) >> rhs);
                } else {
                    return __DPL bit_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) >> rhs);
                }
            },
            arg, shift);
    }

    template <shiftable_type L, simd_abi A, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> arg, R shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        return internal::transform<basic_simd<L, A>>(
            [](auto lhs) {
                if constexpr (integral<L> && sizeof(L) < sizeof(int)) {
                    return static_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) >> R::value);
                } else {
                    return __DPL bit_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) >> R::value);
                }
            },
            arg);
    }

    template <shiftable_type L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> arg, size_t shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        return internal::transform<basic_simd<L, A>>(
            [shift](auto lhs) {
                if constexpr (integral<L> && sizeof(L) < sizeof(int)) {
                    return static_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) >> shift);
                } else {
                    return __DPL bit_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) >> shift);
                }
            },
            arg);
    }

    template <simd_element L, simd_abi A, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd_mask<L, A> arg, R shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        using mask_type = basic_simd_mask<L, A>;
        return []<size_t... Is>(mask_type arg, index_sequence<Is...>) {
            return dx::initialize<mask_type>((
                Is + R::value < simd_abi_traits<L, A>::size ? arg[Is + R::value]
                                                            : false)...);
        }(arg, iota_sequence<L, A>);
    }

    template <simd_element L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd_mask<L, A> arg, size_t shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        using mask_type = basic_simd_mask<L, A>;
        return []<size_t... Is>(
                   mask_type arg, unsigned shift, index_sequence<Is...>) {
            return dx::initialize<mask_type>(
                (Is + shift < simd_abi_traits<L, A>::size ? arg[Is + shift]
                                                          : false)...);
        }(arg, static_cast<unsigned>(shift), iota_sequence<L, A>);
    }

public:
    template <integral_simd R, vshiftable_simd_by<R> L>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R shift) noexcept {
        using A = L::abi_type;
        if constexpr (unqualified_vbwshift_right<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(arg, shift);
                } else {
                    return bwshift_right(internal::abi<A>, arg, shift);
                }
            } else {
                return bwshift_right(internal::abi<A>, arg, shift);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(arg, shift);
        } else {
            return operator()(dx::to_basic_type(arg), dx::to_basic_type(shift));
        }
    }

    template <integral_simd R, vshiftable_simd_by<R> L>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_vbwshift_right<L, R> ||
            unqualified_vbwshift_right<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R shift) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_vbwshift_right<L, R>) {
            return bwshift_right(internal::abi<A>, arg, shift);
        } else {
            return bwshift_right(internal::abi<A>, dx::to_basic_type(arg),
                dx::to_basic_type(shift));
        }
    }

    template <fixed_width_class L, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L arg, R shift) noexcept {
        static_assert(R::value > 0 &&
            R::value < sizeof(typename L::value_type) * char_bit_v);
        if constexpr (unqualified_bwshift_righti<L, R>) {
            if constexpr (basic_simd_class<L>) {
                if consteval {
                    return fallback(arg, shift);
                } else {
                    return bwshift_right<R::value>(internal::abi<L>, arg);
                }
            } else {
                return bwshift_right<R::value>(internal::abi<L>, arg);
            }
        } else if constexpr (basic_simd_class<L>) {
            return fallback(arg, shift);
        } else {
            return operator()(dx::to_basic_type(arg), shift);
        }
    }

    template <scalable_class L, integral_constant_like R>
    requires unqualified_bwshift_righti<L, R> ||
        unqualified_bwshift_righti<basic_type_t<L>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R shift) noexcept {
        static_assert(R::value > 0 &&
            R::value < sizeof(typename L::value_type) * char_bit_v);
        if constexpr (unqualified_bwshift_righti<L, R>) {
            return bwshift_right<R::value>(internal::abi<L>, arg);
        } else {
            return bwshift_right<R::value>(
                internal::abi<L>, dx::to_basic_type(arg));
        }
    }

    template <fixed_width_class L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t shift) noexcept {
        if constexpr (unqualified_bwshift_right<L>) {
            if constexpr (basic_simd_class<L>) {
                if consteval {
                    return fallback(arg, shift);
                } else {
                    return bwshift_right(internal::abi<L>, arg, shift);
                }
            } else {
                return bwshift_right(internal::abi<L>, arg, shift);
            }
        } else if constexpr (basic_simd_class<L>) {
            return fallback(arg, shift);
        } else {
            return operator()(dx::to_basic_type(arg), shift);
        }
    }

    template <scalable_class L>
    requires unqualified_bwshift_right<L> ||
        unqualified_bwshift_right<basic_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t shift) noexcept {
        if constexpr (unqualified_bwshift_right<L>) {
            return bwshift_right(internal::abi<L>, arg, shift);
        } else {
            return bwshift_right(
                internal::abi<L>, dx::to_basic_type(arg), shift);
        }
    }
};

template <size_t V>
struct bwshift_lefti_t {
public:
    template <simd_class T>
    requires regular_invocable<bwshift_left_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return bwshift_left_t::operator()(arg, imm<V>);
    }
};

template <size_t V>
struct bwshift_righti_t {
public:
    template <simd_class T>
    requires regular_invocable<bwshift_right_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return bwshift_right_t::operator()(arg, imm<V>);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bwor_t bwor{};
DPL_EXPORT inline constexpr internal::bwand_t bwand{};
DPL_EXPORT inline constexpr internal::bwxor_t bwxor{};
DPL_EXPORT inline constexpr internal::bwandnot_t bwandnot{};
DPL_EXPORT inline constexpr internal::bwornot_t bwornot{};
DPL_EXPORT inline constexpr internal::bwnot_t bwnot{};
DPL_EXPORT inline constexpr internal::bwshift_left_t bwshift_left{};
DPL_EXPORT inline constexpr internal::bwshift_right_t bwshift_right{};
DPL_EXPORT template <size_t V>
inline constexpr internal::bwshift_lefti_t<V> bwshift_lefti{};
DPL_EXPORT template <size_t V>
inline constexpr internal::bwshift_righti_t<V> bwshift_righti{};
} // namespace cpo

DPL_EXPORT template <typename D>
class bitwise_simd_interface {
public:
    template <typename R>
    requires regular_invocable<internal::bwor_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::bwor_t, D, R> operator|(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::bwor(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwand_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::bwand_t, D, R> operator&(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::bwand(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwxor_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::bwxor_t, D, R> operator^(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::bwxor(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwor_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::bwor_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator|=(this D& lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::bwor(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::bwand_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::bwand_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator&=(this D& lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::bwand(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::bwxor_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::bwxor_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator^=(this D& lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::bwxor(lhs, rhs));
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::bwor_t, L, D> operator|(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::bwor(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::bwand_t, L, D> operator&(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::bwand(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::bwxor_t, L, D> operator^(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::bwxor(lhs, rhs);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr D operator~(this D self) noexcept
    requires simd_type<D> && regular_invocable<internal::bwnot_t, D>
    {
        return datapar::bwnot(self);
    }
};

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwor_t, L, R> operator|(
    L lhs, R rhs) noexcept {
    return datapar::bwor(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwand_t, L, R> operator&(
    L lhs, R rhs) noexcept {
    return datapar::bwand(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwxor_t, L, R> operator^(
    L lhs, R rhs) noexcept {
    return datapar::bwxor(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator|=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs | rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs | rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator&=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs & rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs & rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator^=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs ^ rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs ^ rhs);
}

DPL_EXPORT template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto operator~(T val) noexcept
    -> invoke_result_t<internal::bwnot_t, T> {
    return datapar::bwnot(val);
}

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
