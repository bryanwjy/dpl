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
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
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
template <auto>
void bwshift_left(...) noexcept = delete;
template <auto>
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

    template <simd_element L, common_bits_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
                return __DPL bit_cast<T>(__DPL bit_cast<bit_type>(lhs) |
                    __DPL bit_cast<bit_type>(rhs));
            },
            lhs, rhs);
    }

    template <simd_element L, common_size_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<L, A> left, basic_simd_mask<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            [](bool lhs, bool rhs) { return lhs || rhs; }, left, right);
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bwor<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
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
    requires (!same_abi_simd_as<L, R>) &&
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
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_mbwor<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_mask_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
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
    requires (!same_abi_simd_as<L, R>) &&
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

    template <simd_element L, common_bits_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
                return __DPL bit_cast<T>(__DPL bit_cast<bit_type>(lhs) &
                    __DPL bit_cast<bit_type>(rhs));
            },
            left, right);
    }

    template <simd_element L, common_size_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<L, A> left, basic_simd_mask<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            [](bool lhs, bool rhs) { return lhs && rhs; }, left, right);
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bwand<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
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
    requires (!same_abi_simd_as<L, R>) &&
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
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_mbwand<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_mask_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
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
    requires (!same_abi_simd_as<L, R>) &&
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

    template <simd_element L, common_bits_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
                return __DPL bit_cast<T>(__DPL bit_cast<bit_type>(lhs) ^
                    __DPL bit_cast<bit_type>(rhs));
            },
            left, right);
    }

    template <simd_element L, common_size_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<L, A> left, basic_simd_mask<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            [](bool lhs, bool rhs) { return lhs != rhs; }, left, right);
    }

public:
    template <basic_simd_type L, basic_simd_type R>
    requires common_bits_simd_with<L, R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_bits_type_t<L, R>;
        if constexpr (unqualified_bwxor<T, L, R>) {
            if consteval {
                return fallback(left, right);
            } else {
                return bwxor(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <simd_type L, simd_type R>
    requires common_bits_simd_with<L, R> && only_unqualified<L, R> &&
        unqualified_bwxor<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<bwsimd_for<L, R>> auto {
        return bwxor(internal::abi<common_abi_t<L, R>>, left, right);
    }

    template <simd_type L, simd_type R>
    requires common_bits_simd_with<L, R> &&
        (!basic_simd_type<L> || !basic_simd_type<R>) &&
        (!unqualified_bwxor<common_abi_t<L, R>, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<bwsimd_for<L, R>> auto {
        return operator()(dx::to_basic_type(left), dx::to_basic_type(right));
    }

    template <basic_simd_mask_type L, basic_simd_mask_type R>
    requires common_size_simd_with<L, R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_bits_type_t<L, R>;
        if constexpr (unqualified_bwxor<T, L, R>) {
            if consteval {
                return fallback(left, right);
            } else {
                return bwxor(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <simd_mask_type L, simd_mask_type R>
    requires common_size_simd_with<L, R> && only_unqualified<L, R> &&
        unqualified_bwxor<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_mask_as<bwmask_for<L, R>> auto {
        return bwxor(internal::abi<common_abi_t<L, R>>, left, right);
    }

    template <simd_mask_type L, simd_mask_type R>
    requires common_size_simd_with<L, R> &&
        (!basic_simd_mask_type<L> || !basic_simd_mask_type<R>) &&
        (!unqualified_bwxor<common_abi_t<L, R>, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<bwsimd_for<L, R>> auto {
        return operator()(dx::to_basic_type(left), dx::to_basic_type(right));
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

    template <simd_element L, common_bits_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
                return __DPL bit_cast<T>(__DPL bit_cast<bit_type>(lhs) &
                    ~__DPL bit_cast<bit_type>(rhs));
            },
            left, right);
    }

    template <simd_element L, common_size_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<L, A> left, basic_simd_mask<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            [](bool lhs, bool rhs) { return lhs && !rhs; }, left, right);
    }

public:
    template <basic_simd_type L, basic_simd_type R>
    requires common_bits_simd_with<L, R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_bits_type_t<L, R>;
        if constexpr (unqualified_bwandnot<T, L, R>) {
            if consteval {
                return fallback(left, right);
            } else {
                return bwandnot(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <simd_type L, simd_type R>
    requires common_bits_simd_with<L, R> && only_unqualified<L, R> &&
        unqualified_bwandnot<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<bwsimd_for<L, R>> auto {
        return bwandnot(internal::abi<common_abi_t<L, R>>, left, right);
    }

    template <simd_type L, simd_type R>
    requires common_bits_simd_with<L, R> &&
        (!basic_simd_type<L> || !basic_simd_type<R>) &&
        (!unqualified_bwandnot<common_abi_t<L, R>, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<bwsimd_for<L, R>> auto {
        return operator()(dx::to_basic_type(left), dx::to_basic_type(right));
    }

    template <basic_simd_mask_type L, basic_simd_mask_type R>
    requires common_size_simd_with<L, R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_bits_type_t<L, R>;
        if constexpr (unqualified_bwandnot<T, L, R>) {
            if consteval {
                return fallback(left, right);
            } else {
                return bwandnot(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <simd_mask_type L, simd_mask_type R>
    requires common_size_simd_with<L, R> && only_unqualified<L, R> &&
        unqualified_bwandnot<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_mask_as<bwmask_for<L, R>> auto {
        return bwandnot(internal::abi<common_abi_t<L, R>>, left, right);
    }

    template <simd_mask_type L, simd_mask_type R>
    requires common_size_simd_with<L, R> &&
        (!basic_simd_mask_type<L> || !basic_simd_mask_type<R>) &&
        (!unqualified_bwandnot<common_abi_t<L, R>, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<bwsimd_for<L, R>> auto {
        return operator()(dx::to_basic_type(left), dx::to_basic_type(right));
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

    template <simd_element L, common_bits_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
                return __DPL bit_cast<T>(__DPL bit_cast<bit_type>(lhs) |
                    ~__DPL bit_cast<bit_type>(rhs));
            },
            left, right);
    }

    template <simd_element L, common_size_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd_mask<L, A> left, basic_simd_mask<R, A> right) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            [](bool lhs, bool rhs) { return lhs || !rhs; }, left, right);
    }

public:
    template <basic_simd_type L, basic_simd_type R>
    requires common_bits_simd_with<L, R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_bits_type_t<L, R>;
        if constexpr (unqualified_bwornot<T, L, R>) {
            if consteval {
                return fallback(left, right);
            } else {
                return bwornot(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <simd_type L, simd_type R>
    requires common_bits_simd_with<L, R> && only_unqualified<L, R> &&
        unqualified_bwornot<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<bwsimd_for<L, R>> auto {
        return bwornot(internal::abi<common_abi_t<L, R>>, left, right);
    }

    template <simd_type L, simd_type R>
    requires common_bits_simd_with<L, R> &&
        (!basic_simd_type<L> || !basic_simd_type<R>) &&
        (!unqualified_bwornot<common_abi_t<L, R>, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<bwsimd_for<L, R>> auto {
        return operator()(dx::to_basic_type(left), dx::to_basic_type(right));
    }

    template <basic_simd_mask_type L, basic_simd_mask_type R>
    requires common_size_simd_with<L, R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_bits_type_t<L, R>;
        if constexpr (unqualified_bwornot<T, L, R>) {
            if consteval {
                return fallback(left, right);
            } else {
                return bwornot(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <simd_mask_type L, simd_mask_type R>
    requires common_size_simd_with<L, R> && only_unqualified<L, R> &&
        unqualified_bwornot<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_mask_as<bwmask_for<L, R>> auto {
        return bwornot(internal::abi<common_abi_t<L, R>>, left, right);
    }

    template <simd_mask_type L, simd_mask_type R>
    requires common_size_simd_with<L, R> &&
        (!basic_simd_mask_type<L> || !basic_simd_mask_type<R>) &&
        (!unqualified_bwornot<common_abi_t<L, R>, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<bwsimd_for<L, R>> auto {
        return operator()(dx::to_basic_type(left), dx::to_basic_type(right));
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
                    return __DPL bit_cast<E>(~__DPL bit_cast<bit_type>(arg));
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
    template <basic_simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (requires { bwnot(internal::abi<T>, val); }) {
            if not consteval {
                return bwnot(internal::abi<T>, val);
            } else {
                return fallback(val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T val) noexcept
        -> equivalent_class_as<T> auto {
        if constexpr (requires { bwnot(internal::abi<T>, val); }) {
            return bwnot(internal::abi<T>, val);
        } else {
            return operator()(dx::to_basic_type(val));
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
        fallback(basic_simd<L, A> arg, int shift) noexcept {
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
        fallback(basic_simd_mask<L, A> arg, int shift) noexcept {
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
    requires same_abi_simd_as<L, R>
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
    requires (!same_abi_simd_as<L, R>) &&
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

    template <simd_class L, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
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

    template <simd_class L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L arg, int shift) noexcept {
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
        fallback(basic_simd<L, A> arg, int shift) noexcept {
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
            return dx::initialize<mask_type>(
                (Is + R::value < element_count<L, A> ? arg[Is + R::value]
                                                     : false)...);
        }(arg, iota_sequence<L, A>);
    }

    template <simd_element L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd_mask<L, A> arg, int shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        using mask_type = basic_simd_mask<L, A>;
        return []<size_t... Is>(
                   mask_type arg, unsigned shift, index_sequence<Is...>) {
            return dx::initialize<mask_type>((
                Is + shift < element_count<L, A> ? arg[Is + shift] : false)...);
        }(arg, static_cast<unsigned>(shift), iota_sequence<L, A>);
    }

public:
    template <integral_simd R, vshiftable_simd_by<R> L>
    requires same_abi_simd_as<L, R>
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
    requires (!same_abi_simd_as<L, R>) &&
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

    template <simd_class L, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
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

    template <simd_class L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L arg, int shift) noexcept {
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
