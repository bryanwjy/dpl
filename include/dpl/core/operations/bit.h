// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/select.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/common_bits_with.h"
#  include "dpl/core/concepts/compatible_mask_with.h"
#  include "dpl/core/concepts/immediate_mask_like.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/all_bits.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/common_bits_type.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/std/bit/byteswap.h"
#  include "dpl/std/bit/countl.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bit_drop(...) noexcept = delete;
template <auto>
void bit_drop(...) noexcept = delete;
void bit_fill(...) noexcept = delete;
template <auto>
void bit_fill(...) noexcept = delete;
void bit_keep(...) noexcept = delete;
template <auto>
void bit_keep(...) noexcept = delete;
void bit_stencil(...) noexcept = delete;
template <auto>
void bit_stencil(...) noexcept = delete;
template <auto>
void bit_select(...) noexcept = delete;
void bit_select(...) noexcept = delete;
void popcount(...) noexcept = delete;
void countl_zero(...) noexcept = delete;
void countl_one(...) noexcept = delete;
void countr_zero(...) noexcept = delete;
void countr_one(...) noexcept = delete;
void byteswap(...) noexcept = delete;

template <typename T, typename L, typename R>
concept bit_result = common_bits_simd_with<T, common_bits_simd_t<L, R>> &&
    same_abi_as<common_abi_t<L, R>, typename T::abi_type>;

template <typename T, typename L, typename R>
concept mbit_result = simd_mask_type<L> && common_size_simd_with<T, R> &&
    same_abi_as<common_abi_t<L, R>, typename T::abi_type>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bit_drop = requires(L lhs, R rhs) {
    { bit_drop(internal::abi<A>, lhs, rhs) } -> bit_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mbit_drop = requires(L lhs, R rhs) {
    { bit_drop(internal::abi<A>, lhs, rhs) } -> mbit_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bit_fill = requires(L lhs, R rhs) {
    { bit_fill(internal::abi<A>, lhs, rhs) } -> bit_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mbit_fill = requires(L lhs, R rhs) {
    { bit_fill(internal::abi<A>, lhs, rhs) } -> mbit_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bit_keep =
    requires(L lhs, R rhs) { bit_keep(internal::abi<A>, lhs, rhs); };

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mbit_keep = requires(L lhs, R rhs) {
    { bit_keep(internal::abi<A>, lhs, rhs) } -> mbit_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bit_stencil =
    requires(L lhs, R rhs) { bit_stencil(internal::abi<A>, lhs, rhs); };

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mbit_stencil = requires(L lhs, R rhs) {
    { bit_stencil(internal::abi<A>, lhs, rhs) } -> mbit_result<L, R>;
};

template <typename T, typename C, typename L, typename R>
concept bit_result3 = common_bits_simd_with<T, common_bits_simd_t<L, R, C>> &&
    same_abi_as<common_abi_t<L, R, C>, typename T::abi_type>;

template <typename T, typename C, typename L, typename R>
concept mbit_result3 = mbit_result<T, C, L> && mbit_result<T, C, R> &&
    same_abi_as<common_abi_t<L, R, C>, typename T::abi_type>;

template <typename C, typename L, typename R,
    typename A = common_abi_t<C, L, R>>
concept unqualified_bit_select = requires(C cond, L lhs, R rhs) {
    { bit_select(internal::abi<A>, cond, lhs, rhs) } -> bit_result3<C, L, R>;
};

template <typename C, typename L, typename R,
    typename A = common_abi_t<C, L, R>>
concept unqualified_mbit_select = requires(C cond, L lhs, R rhs) {
    { bit_select(internal::abi<A>, cond, lhs, rhs) } -> mbit_result3<C, L, R>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bit_selecti =
    immediate_mask_for<M, L> && requires(L lhs, R rhs) {
        {
            bit_select<immediate_mask_v<L, M>>(internal::abi<A>, lhs, rhs)
        } -> bit_result<L, R>;
    };

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mbit_selecti =
    immediate_mask_for<M, L> && requires(L lhs, R rhs) {
        {
            bit_select<immediate_mask_v<L, M>>(internal::abi<A>, lhs, rhs)
        } -> mbit_result<L, R>;
    };

struct bit_drop_t {
private:
    template <typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(M mask, T arg) noexcept {
        if constexpr (atom::common_class_with<M, T>) {
            return dx::bwandnot(arg, mask);
        } else if constexpr (basic_simd_type<T>) {
            return dx::select(mask, dx::zero, arg);
        } else {
            return dx::select(mask, dx::broadcast<T>(false), arg);
        }
    }

    template <auto V, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::selecti<V>(dx::zero, arg);
        } else {
            return dx::selecti<V>(dx::broadcast<T>(false), arg);
        }
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bit_drop<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return bit_drop(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bit_drop(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_bits_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_bit_drop<L, R> ||
            unqualified_bit_drop<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_bit_drop<L, R>) {
            return bit_drop(internal::abi<A>, lhs, rhs);
        } else {
            return bit_drop(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_type R, compatible_mask_with<R> L>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_mbit_drop<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return bit_drop(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bit_drop(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_mask_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type R, compatible_mask_with<R> L>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mbit_drop<L, R> ||
            unqualified_mbit_drop<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mbit_drop<L, R>) {
            return bit_drop(internal::abi<A>, lhs, rhs);
        } else {
            return bit_drop(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_class R, immediate_mask_for<R> L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<R>(lhs))::value;
        if constexpr (requires {
                          {
                              bit_drop<V>(internal::abi<R>, rhs)
                          } -> equivalent_class_as<R>;
                      }) {
            if constexpr (basic_simd_class<R>) {
                if consteval {
                    return fallbacki<V>(rhs);
                } else {
                    return bit_drop<V>(internal::abi<R>, rhs);
                }
            } else {
                return bit_drop<V>(internal::abi<R>, rhs);
            }
        } else if constexpr (basic_simd_class<R>) {
            return fallbacki<V>(rhs);
        } else {
            return operator()(dx::to_basic_type(rhs));
        }
    }

    template <simd_class L, broadcastable_to<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (requires {
                          {
                              bit_drop(internal::abi<L>, lhs, rhs)
                          } -> equivalent_simd_as<make_simd_type_t<L>>;
                      }) {
            if constexpr (basic_simd_class<L>) {
                if consteval {
                    return operator()(lhs, dx::broadcast<L>(rhs));
                } else {
                    return bit_drop(internal::abi<L>, lhs, rhs);
                }
            } else {
                return bit_drop(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (basic_simd_class<R>) {
            return operator()(dx::broadcast<R>(lhs), rhs);
        } else {
            return operator()(lhs, dx::to_basic_type(rhs));
        }
    }
};

template <auto V>
struct bit_dropi_t {
private:
    template <typename T>
    using imm_mask DPL_NODEBUG = make_immediate_mask_t<T, V>;

public:
    template <simd_class T>
    requires requires {
        typename imm_mask<T>;
        requires regular_invocable<bit_drop_t, imm_mask<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        constexpr imm_mask<T> mask{};
        return bit_drop_t::operator()(mask, arg);
    }
};

struct bit_fill_t {
private:
    template <typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(M mask, T arg) noexcept {
        if constexpr (atom::common_class_with<M, T>) {
            return dx::bwor(mask, arg);
        } else if constexpr (basic_simd_type<T>) {
            return dx::select(mask, dx::all_bits, arg);
        } else {
            return dx::select(mask, dx::broadcast<T>(true), arg);
        }
    }

    template <auto V, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::selecti<V>(dx::all_bits, arg);
        } else {
            return dx::selecti<V>(dx::broadcast<T>(true), arg);
        }
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bit_fill<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return bit_fill(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bit_fill(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_bits_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_bit_fill<L, R> ||
            unqualified_bit_fill<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_bit_fill<L, R>) {
            return bit_fill(internal::abi<A>, lhs, rhs);
        } else {
            return bit_fill(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_type R, compatible_mask_with<R> L>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_mbit_fill<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return bit_fill(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bit_fill(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_mask_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type R, compatible_mask_with<R> L>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mbit_fill<L, R> ||
            unqualified_mbit_fill<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mbit_fill<L, R>) {
            return bit_fill(internal::abi<A>, lhs, rhs);
        } else {
            return bit_fill(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_class R, immediate_mask_for<R> L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<R>(lhs))::value;
        if constexpr (requires {
                          {
                              bit_fill<V>(internal::abi<R>, rhs)
                          } -> equivalent_class_as<R>;
                      }) {
            if constexpr (basic_simd_class<R>) {
                if consteval {
                    return fallbacki<V>(rhs);
                } else {
                    return bit_fill<V>(internal::abi<R>, rhs);
                }
            } else {
                return bit_fill<V>(internal::abi<R>, rhs);
            }
        } else if constexpr (basic_simd_class<R>) {
            return fallbacki<V>(rhs);
        } else {
            return operator()(dx::to_basic_type(rhs));
        }
    }

    template <simd_class L, broadcastable_to<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (requires {
                          {
                              bit_fill(internal::abi<L>, lhs, rhs)
                          } -> equivalent_simd_as<make_simd_type_t<L>>;
                      }) {
            if constexpr (basic_simd_class<L>) {
                if consteval {
                    return operator()(lhs, dx::broadcast<L>(rhs));
                } else {
                    return bit_fill(internal::abi<L>, lhs, rhs);
                }
            } else {
                return bit_fill(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (basic_simd_class<R>) {
            return operator()(dx::broadcast<R>(lhs), rhs);
        } else {
            return operator()(lhs, dx::to_basic_type(rhs));
        }
    }
};

template <auto V>
struct bit_filli_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_immediate_mask_t<T, V>;

public:
    template <simd_class T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<bit_fill_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return bit_fill_t::operator()(mask, arg);
    }
};

struct bit_keep_t {
private:
    template <typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(M mask, T arg) noexcept {
        if constexpr (atom::common_class_with<M, T>) {
            return dx::bwand(mask, arg);
        } else if constexpr (basic_simd_type<T>) {
            return dx::select(mask, arg, dx::zero);
        } else {
            return dx::select(mask, arg, dx::broadcast<T>(false));
        }
    }

    template <auto V, basic_simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::selecti<V>(arg, dx::zero);
        } else {
            return dx::selecti<V>(arg, dx::broadcast<T>(false));
        }
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bit_keep<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return bit_keep(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bit_keep(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_bits_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_bit_keep<L, R> ||
            unqualified_bit_keep<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_bit_keep<L, R>) {
            return bit_keep(internal::abi<A>, lhs, rhs);
        } else {
            return bit_keep(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_type R, compatible_mask_with<R> L>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_mbit_keep<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return bit_keep(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bit_keep(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_mask_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type R, compatible_mask_with<R> L>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mbit_keep<L, R> ||
            unqualified_mbit_keep<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mbit_keep<L, R>) {
            return bit_keep(internal::abi<A>, lhs, rhs);
        } else {
            return bit_keep(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_class R, immediate_mask_for<R> L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<R>(lhs))::value;
        if constexpr (requires {
                          {
                              bit_keep<V>(internal::abi<R>, rhs)
                          } -> equivalent_class_as<R>;
                      }) {
            if constexpr (basic_simd_class<R>) {
                if consteval {
                    return fallbacki<V>(rhs);
                } else {
                    return bit_keep<V>(internal::abi<R>, rhs);
                }
            } else {
                return bit_keep<V>(internal::abi<R>, rhs);
            }
        } else if constexpr (basic_simd_class<R>) {
            return fallbacki<V>(rhs);
        } else {
            return operator()(dx::to_basic_type(rhs));
        }
    }

    template <simd_class L, broadcastable_to<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (requires {
                          {
                              bit_keep(internal::abi<L>, lhs, rhs)
                          } -> equivalent_simd_as<make_simd_type_t<L>>;
                      }) {
            if constexpr (basic_simd_class<L>) {
                if consteval {
                    return operator()(lhs, dx::broadcast<L>(rhs));
                } else {
                    return bit_keep(internal::abi<L>, lhs, rhs);
                }
            } else {
                return bit_keep(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (basic_simd_class<R>) {
            return operator()(dx::broadcast<R>(lhs), rhs);
        } else {
            return operator()(lhs, dx::to_basic_type(rhs));
        }
    }
};

template <auto V>
struct bit_keepi_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_immediate_mask_t<T, V>;

public:
    template <simd_class T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<bit_keep_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return bit_keep_t::operator()(mask, arg);
    }
};

struct bit_stencil_t {
private:
    template <typename M, typename T, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<M, A> mask, basic_simd<T, A> arg) noexcept {
        return dx::bwornot(arg, mask);
    }

    template <typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(M mask, T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::select(mask, arg, dx::all_bits);
        } else {
            return dx::select(mask, arg, dx::broadcast<T>(true));
        }
    }

    template <auto V, basic_simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::selecti<V>(arg, dx::all_bits);
        } else {
            return dx::selecti<V>(arg, dx::broadcast<T>(true));
        }
    }

public:
    template <simd_type L, common_bits_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_bit_stencil<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return bit_stencil(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bit_stencil(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_bits_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_bit_stencil<L, R> ||
            unqualified_bit_stencil<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_bit_stencil<L, R>) {
            return bit_stencil(internal::abi<A>, lhs, rhs);
        } else {
            return bit_stencil(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_type R, compatible_mask_with<R> L>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_mbit_stencil<L, R, A>) {
            if constexpr (basic_simd_mask_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return bit_stencil(internal::abi<A>, lhs, rhs);
                }
            } else {
                return bit_stencil(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_mask_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type R, compatible_mask_with<R> L>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mbit_stencil<L, R> ||
            unqualified_mbit_stencil<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mbit_stencil<L, R>) {
            return bit_stencil(internal::abi<A>, lhs, rhs);
        } else {
            return bit_stencil(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_class R, immediate_mask_for<R> L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<R>(lhs))::value;
        if constexpr (requires {
                          {
                              bit_stencil<V>(internal::abi<R>, rhs)
                          } -> equivalent_class_as<R>;
                      }) {
            if constexpr (basic_simd_class<R>) {
                if consteval {
                    return fallbacki<V>(rhs);
                } else {
                    return bit_stencil<V>(internal::abi<R>, rhs);
                }
            } else {
                return bit_stencil<V>(internal::abi<R>, rhs);
            }
        } else if constexpr (basic_simd_class<R>) {
            return fallbacki<V>(rhs);
        } else {
            return operator()(dx::to_basic_type(rhs));
        }
    }

    template <simd_class L, broadcastable_to<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (requires {
                          {
                              bit_stencil(internal::abi<L>, lhs, rhs)
                          } -> equivalent_simd_as<make_simd_type_t<L>>;
                      }) {
            if constexpr (basic_simd_class<L>) {
                if consteval {
                    return operator()(lhs, dx::broadcast<L>(rhs));
                } else {
                    return bit_stencil(internal::abi<L>, lhs, rhs);
                }
            } else {
                return bit_stencil(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (basic_simd_class<R>) {
            return operator()(dx::broadcast<R>(lhs), rhs);
        } else {
            return operator()(lhs, dx::to_basic_type(rhs));
        }
    }
};

template <auto V>
struct bit_stencili_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_immediate_mask_t<T, V>;

public:
    template <simd_class T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<bit_stencil_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return bit_stencil_t::operator()(mask, arg);
    }
};

struct bit_select_t {
private:
    template <auto V, typename ET, typename EF, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallbacki(basic_simd<ET, A> tval, basic_simd<EF, A> fval) noexcept {
        using ER = common_bits_type_t<ET, EF>;
        return dx::selecti<V>(
            dx::reinterpret<ER>(tval), dx::reinterpret<ER>(fval));
    }

    template <auto V, typename ET, typename EF, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_simd_mask<ET, A> tval, basic_simd_mask<EF, A> fval) noexcept {
        using ER = common_size_type_t<ET, EF>;
        return dx::selecti<V>(
            dx::reinterpret<ER>(tval), dx::reinterpret<ER>(fval));
    }

    template <typename EM, typename ET, typename EF, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_simd_mask<EM, A> mask,
        basic_simd<ET, A> tval, basic_simd<EF, A> fval) noexcept {
        using ER = common_bits_type_t<ET, EF>;
        return dx::select(
            mask, dx::reinterpret<ER>(tval), dx::reinterpret<ER>(fval));
    }

    template <typename EM, typename ET, typename EF, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_simd_mask<EM, A> mask,
        basic_simd_mask<ET, A> tval, basic_simd_mask<EF, A> fval) noexcept {
        using ER = common_size_type_t<ET, EF>;
        return dx::select(
            mask, dx::reinterpret<ER>(tval), dx::reinterpret<ER>(fval));
    }

    template <typename EM, typename ET, typename EF, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_simd<EM, A> mask,
        basic_simd<ET, A> tval, basic_simd<EF, A> fval) noexcept {
        using ER = common_bits_type_t<ET, EF, EM>;
        return dx::bwor(
            dx::bwand(dx::reinterpret<ER>(mask), dx::reinterpret<ER>(tval)),
            dx::bwandnot(dx::reinterpret<ER>(fval), dx::reinterpret<ER>(mask)));
    }

    template <typename T, typename F, typename M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto impli(M mask, T tval, F fval) {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        using A = T::abi_type;
        if constexpr (unqualified_bit_selecti<M, T, F, A> ||
            unqualified_mbit_selecti<M, T, F, A>) {
            if constexpr (basic_simd_class<T> && basic_simd_class<F>) {
                if consteval {
                    return fallbacki<V>(tval, fval);
                } else {
                    return bit_select<V>(internal::abi<T>, tval, fval);
                }
            } else {
                return bit_select<V>(internal::abi<T>, tval, fval);
            }
        } else if constexpr (basic_simd_class<T> && basic_simd_class<F>) {
            return fallbacki<V>(tval, fval);
        } else {
            return impli(
                mask, dx::to_basic_type(tval), dx::to_basic_type(fval));
        }
    }

    template <typename M, typename T, typename F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto impl(M mask, T tval, F fval) {
        using A = T::abi_type;
        if constexpr (unqualified_bit_select<M, T, F, A> ||
            unqualified_mbit_select<M, T, F, A>) {
            if constexpr (basic_simd_class<T> && basic_simd_class<F>) {
                if consteval {
                    return fallback(mask, tval, fval);
                } else {
                    return bit_select(internal::abi<T>, mask, tval, fval);
                }
            } else {
                return bit_select(internal::abi<T>, mask, tval, fval);
            }
        } else if constexpr (basic_simd_class<T> && basic_simd_class<F>) {
            return fallback(mask, tval, fval);
        } else {
            return impl(mask, dx::to_basic_type(tval), dx::to_basic_type(fval));
        }
    }

public:
    template <simd_type T, common_bits_simd_with<T> F, immediate_mask_for<T> M>
    requires same_abi_simd_as<T, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T tval, F fval) noexcept {
        return impli(mask, tval, fval);
    }

    template <simd_type T, common_bits_simd_with<T> F, immediate_mask_for<T> M>
    requires (!same_abi_simd_as<T, F>) &&
        (unqualified_bit_selecti<M, T, F> ||
            unqualified_bit_selecti<M, basic_type_t<T>, basic_type_t<F>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T tval, F fval) noexcept {
        using A = common_abi_t<T, F>;
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (unqualified_bit_selecti<M, T, F>) {
            return bit_select<V>(internal::abi<A>, tval, fval);
        } else {
            return bit_select<V>(internal::abi<A>, dx::to_basic_type(tval),
                dx::to_basic_type(fval));
        }
    }

    template <simd_type T, common_bits_simd_with<T> F,
        compatible_mask_with<common_bits_simd_t<T, F>> M>
    requires same_abi_simd_as<T, F> && same_abi_simd_as<T, M> &&
        same_abi_simd_as<F, M> && fixed_width_abi<common_abi_t<T, F, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T tval, F fval) noexcept {
        return impl(mask, tval, fval);
    }

    template <simd_type T, common_bits_simd_with<T> F,
        compatible_mask_with<common_bits_simd_t<T, F>> M>
    requires (!same_abi_simd_as<T, F> || !same_abi_simd_as<T, M> ||
                 !same_abi_simd_as<F, M> ||
                 scalable_abi<common_abi_t<T, F, M>>) &&
        (unqualified_bit_select<M, T, F> ||
            unqualified_bit_select<M, basic_type_t<T>, basic_type_t<F>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T tval, F fval) noexcept {
        using A = common_abi_t<T, F, M>;
        if constexpr (unqualified_bit_select<M, T, F>) {
            return bit_select(internal::abi<A>, mask, tval, fval);
        } else {
            return bit_select(internal::abi<A>, dx::to_basic_type(mask),
                dx::to_basic_type(tval), dx::to_basic_type(fval));
        }
    }
};

template <auto V>
struct bit_selecti_t : binary_operation_base<bit_selecti_t<V>> {
private:
    friend binary_operation_base<bit_selecti_t>;

    template <typename T>
    using mask_type DPL_NODEBUG = make_immediate_mask_t<T, V>;

    template <simd_abi A, simd_class L, typename R>
    requires unqualified_bit_selecti<mask_type<L>, L, R, A> ||
        unqualified_mbit_selecti<mask_type<L>, L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        native(A abi, L left, R right) noexcept {
        return bit_select<V>(internal::abi<A>, left, right);
    }

    template <simd_abi A, typename L, simd_class R>
    requires unqualified_bit_selecti<mask_type<R>, L, R, A> ||
        unqualified_mbit_selecti<mask_type<R>, L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        native(A abi, L left, R right) noexcept {
        return bit_select<V>(internal::abi<A>, left, right);
    }

public:
    template <simd_class T, simd_class F>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<bit_select_t, mask_type<T>, T, F>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T tval, F fval) noexcept {
        constexpr mask_type<T> mask{};
        return bit_select_t::operator()(mask, tval, fval);
    }

    using binary_operation_base<bit_selecti_t>::operator();
};

template <typename T, typename U>
concept bit_result_simd_for = integral_simd<T> && common_size_simd_with<T, U>;

template <typename T>
concept unqualified_popcount = requires(T arg) {
    { popcount(internal::abi<T>, arg) } -> bit_result_simd_for<T>;
};

template <typename T>
concept unqualified_mpopcount = requires(T arg) {
    { popcount(internal::abi<T>, arg) } -> core_convertible_to<size_t>;
};

struct popcount_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return internal::transform<basic_simd<E, A>>(
            [](auto val) {
                auto const count = __DPL popcount(__DPL to_unsigned(val));
                return static_cast<E>(count);
            },
            arg);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr size_t DPL_VECTORCALL
        fallback(basic_simd_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            if constexpr (I > 0) {
                return arg[I] + self(arg, imm<I - 1>);
            } else {
                return arg[I];
            }
        }(arg, imm<simd_abi_traits<E, A>::size - 1>);
    }

public:
    template <integral_simd T>
    requires fixed_width_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_popcount<T>) {
            if constexpr (basic_simd_type<T>) {
                using E = typename decltype(popcount(
                    internal::abi<T>, arg))::value_type;
                if consteval {
                    return dx::reinterpret<E>(fallback(arg));
                } else {
                    return popcount(internal::abi<T>, arg);
                }
            } else {
                return popcount(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_simd T>
    requires scalable_simd<T> &&
        (unqualified_popcount<T> || unqualified_popcount<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_popcount<T>) {
            return popcount(internal::abi<T>, arg);
        } else {
            return popcount(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mpopcount<T>) {
            if constexpr (basic_simd_mask_type<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return popcount(internal::abi<T>, arg);
                }
            } else {
                return popcount(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_mask_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T>
    requires scalable_mask<T> &&
        (unqualified_mpopcount<T> || unqualified_mpopcount<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mpopcount<T>) {
            return popcount(internal::abi<T>, arg);
        } else {
            return popcount(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    requires unsigned_integral<typename T::value_type>
    static consteval size_t operator()(T) noexcept {
        return __DPL popcount(T::value);
    }

    template <immediate_mask_like T>
    static consteval size_t operator()(T val) noexcept {
        if constexpr (requires {
                          { popcount(val) } -> core_convertible_to<size_t>;
                      }) {
            return popcount(val);
        } else {
            return __DPL popcount(T::value);
        }
    }
};

template <typename T>
concept unqualified_countl_zero = requires(T arg) {
    { countl_zero(internal::abi<T>, arg) } -> bit_result_simd_for<T>;
};

template <typename T>
concept unqualified_mcountl_zero = requires(T arg) {
    { countl_zero(internal::abi<T>, arg) } -> core_convertible_to<size_t>;
};

struct countl_zero_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return internal::transform<basic_simd<E, A>>(arg, [](auto val) {
            auto const count = __DPL countl_zero(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t fallback(basic_simd_mask<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            auto val = !arg[I];
            if constexpr (I > 0) {
                auto const mask = val ? -1 : 0;
                return val + (mask & self(arg, imm<I - 1>));
            } else {
                return val;
            }
        }(arg, imm<simd_abi_traits<E, A>::size - 1>);
    }

public:
    template <integral_simd T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_countl_zero<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    using E = typename decltype(countl_zero(
                        internal::abi<T>, arg))::value_type;
                    return dx::reinterpret<E>(fallback(arg));
                } else {
                    return countl_zero(internal::abi<T>, arg);
                }
            } else {
                return countl_zero(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_simd T>
    requires scalable_simd<T> &&
        (unqualified_countl_zero<T> || unqualified_countl_zero<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_countl_zero<T>) {
            return countl_zero(internal::abi<T>, arg);
        } else {
            return countl_zero(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountl_zero<T>) {
            if constexpr (basic_simd_mask_type<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return countl_zero(internal::abi<T>, arg);
                }
            } else {
                return countl_zero(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_mask_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T>
    requires scalable_mask<T> &&
        (unqualified_mcountl_zero<T> ||
            unqualified_mcountl_zero<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountl_zero<T>) {
            return countl_zero(internal::abi<T>, arg);
        } else {
            return countl_zero(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    requires unsigned_integral<typename T::value_type>
    static consteval size_t operator()(T) noexcept {
        return __DPL countl_zero(T::value);
    }

    template <immediate_mask_like T>
    static consteval size_t operator()(T val) noexcept {
        if constexpr (requires {
                          { countl_zero(val) } -> core_convertible_to<size_t>;
                      }) {
            return countl_zero(val);
        } else {
            constexpr auto mask =
                static_cast<typename T::value_type>((1ll << T::width) - 1);
            constexpr auto remainder =
                sizeof(typename T::value_type) * char_bit_v - T::width;
            return __DPL countl_zero(T::value & mask) - remainder;
        }
    }
};

template <typename T>
concept unqualified_countl_one = requires(T arg) {
    { countl_one(internal::abi<T>, arg) } -> bit_result_simd_for<T>;
};

template <typename T>
concept unqualified_mcountl_one = requires(T arg) {
    { countl_one(internal::abi<T>, arg) } -> core_convertible_to<size_t>;
};

struct countl_one_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return internal::transform<basic_simd<E, A>>(arg, [](auto val) {
            auto const count = __DPL countl_one(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd_mask<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            auto val = arg[I];
            if constexpr (I > 0) {
                auto const mask = val ? -1 : 0;
                return val + (mask & self(arg, imm<I - 1>));
            } else {
                return val;
            }
        }(arg, imm<simd_abi_traits<E, A>::size - 1>);
    }

public:
    template <integral_simd T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_countl_one<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    using E = typename decltype(countl_one(
                        internal::abi<T>, arg))::value_type;
                    return dx::reinterpret<E>(fallback(arg));
                } else {
                    return countl_one(internal::abi<T>, arg);
                }
            } else {
                return countl_one(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_simd T>
    requires scalable_simd<T> &&
        (unqualified_countl_one<T> || unqualified_countl_one<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_countl_one<T>) {
            return countl_one(internal::abi<T>, arg);
        } else {
            return countl_one(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountl_one<T>) {
            if constexpr (basic_simd_mask_type<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return countl_one(internal::abi<T>, arg);
                }
            } else {
                return countl_one(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_mask_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T>
    requires scalable_mask<T> &&
        (unqualified_mcountl_one<T> || unqualified_mcountl_one<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountl_one<T>) {
            return countl_one(internal::abi<T>, arg);
        } else {
            return countl_one(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    requires unsigned_integral<typename T::value_type>
    static consteval size_t operator()(T) noexcept {
        return __DPL countl_one(T::value);
    }

    template <immediate_mask_like T>
    static consteval size_t operator()(T val) noexcept {
        if constexpr (requires {
                          { countl_one(val) } -> core_convertible_to<size_t>;
                      }) {
            return countl_one(val);
        } else {
            constexpr auto mask =
                static_cast<typename T::value_type>(-1ll << T::width);
            constexpr auto remainder =
                sizeof(typename T::value_type) * char_bit_v - T::width;
            return __DPL countl_one(T::value | mask) - remainder;
        }
    }
};

template <typename T>
concept unqualified_countr_zero = requires(T arg) {
    { countr_zero(internal::abi<T>, arg) } -> bit_result_simd_for<T>;
};

template <typename T>
concept unqualified_mcountr_zero = requires(T arg) {
    { countr_zero(internal::abi<T>, arg) } -> core_convertible_to<size_t>;
};

struct countr_zero_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd<E, A> arg) noexcept {
        return internal::transform<basic_simd<E, A>>(arg, [](auto val) {
            auto const count = __DPL countr_zero(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            auto val = !arg[I];
            if constexpr (I < simd_abi_traits<E, A>::size) {
                auto const mask = val ? -1 : 0;
                return val + (mask & self(arg, imm<I - 1>));
            } else {
                return 0;
            }
        }(arg, imm<0>);
    }

public:
    template <integral_simd T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_countr_zero<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    using E = typename decltype(countr_zero(
                        internal::abi<T>, arg))::value_type;
                    return dx::reinterpret<E>(fallback(arg));
                } else {
                    return countr_zero(internal::abi<T>, arg);
                }
            } else {
                return countr_zero(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_simd T>
    requires scalable_simd<T> &&
        (unqualified_countr_zero<T> || unqualified_countr_zero<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_countr_zero<T>) {
            return countr_zero(internal::abi<T>, arg);
        } else {
            return countr_zero(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountr_zero<T>) {
            if constexpr (basic_simd_mask_type<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return countr_zero(internal::abi<T>, arg);
                }
            } else {
                return countr_zero(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_mask_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T>
    requires scalable_mask<T> &&
        (unqualified_mcountr_zero<T> ||
            unqualified_mcountr_zero<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountr_zero<T>) {
            return countr_zero(internal::abi<T>, arg);
        } else {
            return countr_zero(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    requires unsigned_integral<typename T::value_type>
    static consteval size_t operator()(T) noexcept {
        return __DPL countr_zero(T::value);
    }

    template <immediate_mask_like T>
    static consteval size_t operator()(T val) noexcept {
        if constexpr (requires {
                          { countr_zero(val) } -> core_convertible_to<size_t>;
                      }) {
            return countr_zero(val);
        } else {
            auto const count = __DPL countr_zero(T::value);
            return count < T::width ? count : T::width;
        }
    }
};

template <typename T>
concept unqualified_countr_one = requires(T arg) {
    { countr_one(internal::abi<T>, arg) } -> bit_result_simd_for<T>;
};

template <typename T>
concept unqualified_mcountr_one = requires(T arg) {
    { countr_one(internal::abi<T>, arg) } -> core_convertible_to<size_t>;
};

struct countr_one_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> arg) noexcept {
        return internal::transform<basic_simd<E, A>>(arg, [](auto val) {
            auto const count = __DPL countr_one(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            auto val = arg[I];
            if constexpr (I < simd_abi_traits<E, A>::size) {
                auto const mask = val ? -1 : 0;
                return val + (mask & self(arg, imm<I - 1>));
            } else {
                return 0;
            }
        }(arg, imm<0>);
    }

public:
    template <integral_simd T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_countr_one<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    using E = typename decltype(countr_one(
                        internal::abi<T>, arg))::value_type;
                    return dx::reinterpret<E>(fallback(arg));
                } else {
                    return countr_one(internal::abi<T>, arg);
                }
            } else {
                return countr_one(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_simd T>
    requires scalable_simd<T> &&
        (unqualified_countr_one<T> || unqualified_countr_one<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_countr_one<T>) {
            return countr_one(internal::abi<T>, arg);
        } else {
            return countr_one(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountr_one<T>) {
            if constexpr (basic_simd_mask_type<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return countr_one(internal::abi<T>, arg);
                }
            } else {
                return countr_one(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_mask_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T>
    requires scalable_mask<T> &&
        (unqualified_mcountr_one<T> || unqualified_mcountr_one<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountr_one<T>) {
            return countr_one(internal::abi<T>, arg);
        } else {
            return countr_one(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    requires unsigned_integral<typename T::value_type>
    static consteval size_t operator()(T) noexcept {
        return __DPL countr_one(T::value);
    }

    template <immediate_mask_like T>
    static consteval size_t operator()(T val) noexcept {
        if constexpr (requires {
                          { countr_one(val) } -> core_convertible_to<size_t>;
                      }) {
            return countr_one(val);
        } else {
            auto const count = __DPL countr_one(T::value);
            return count < T::width ? count : T::width;
        }
    }
};

template <typename T>
concept unqualified_byteswap = requires(T arg) {
    { byteswap(internal::abi<T>, arg) } -> equivalent_simd_as<T>;
};

struct byteswap_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return internal::transform<basic_simd<E, A>>(
            arg, [](auto val) { return __DPL byteswap(val); });
    }

public:
    template <integral_simd T>
    requires fixed_width_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_byteswap<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return byteswap(internal::abi<T>, arg);
                }
            } else {
                return byteswap(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_simd T>
    requires scalable_simd<T> &&
        (unqualified_byteswap<T> || unqualified_byteswap<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_byteswap<T>) {
            return byteswap(internal::abi<T>, arg);
        } else {
            return byteswap(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    static consteval auto operator()(T) noexcept {
        return __DPL byteswap(T::value);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bit_drop_t bit_drop{};
DPL_EXPORT inline constexpr internal::bit_fill_t bit_fill{};
DPL_EXPORT inline constexpr internal::bit_keep_t bit_keep{};
DPL_EXPORT inline constexpr internal::bit_stencil_t bit_stencil{};
DPL_EXPORT inline constexpr internal::bit_select_t bit_select{};
DPL_EXPORT inline constexpr internal::popcount_t popcount{};
DPL_EXPORT inline constexpr internal::countl_zero_t countl_zero{};
DPL_EXPORT inline constexpr internal::countl_one_t countl_one{};
DPL_EXPORT inline constexpr internal::countr_zero_t countr_zero{};
DPL_EXPORT inline constexpr internal::countr_one_t countr_one{};
DPL_EXPORT inline constexpr internal::byteswap_t byteswap{};
DPL_EXPORT template <auto V>
inline constexpr internal::bit_dropi_t<V> bit_dropi{};
DPL_EXPORT template <auto V>
inline constexpr internal::bit_filli_t<V> bit_filli{};
DPL_EXPORT template <auto V>
inline constexpr internal::bit_keepi_t<V> bit_keepi{};
DPL_EXPORT template <auto V>
inline constexpr internal::bit_stencili_t<V> bit_stencili{};
DPL_EXPORT template <auto V>
inline constexpr internal::bit_selecti_t<V> bit_selecti{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
