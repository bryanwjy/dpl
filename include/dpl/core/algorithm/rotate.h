// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/slide.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/common_order_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <size_t>
void rotate_right(...) noexcept = delete;
void rotate_right(...) noexcept = delete;
template <size_t>
void rotate_left(...) noexcept = delete;
void rotate_left(...) noexcept = delete;

template <size_t V, typename L>
concept unqualified_rotate_righti = requires(L val) {
    { rotate_right<V>(internal::abi<L>, val) } -> equivalent_class_as<L>;
};

template <typename T>
concept unqualified_rotate_right = requires(T val, size_t lanes) {
    { rotate_right(internal::abi<T>, val, lanes) } -> equivalent_class_as<T>;
};

template <size_t V, typename L>
concept unqualified_rotate_lefti = requires(L val) {
    { rotate_left<V>(internal::abi<L>, val) } -> equivalent_class_as<L>;
};

template <typename T>
concept unqualified_rotate_left = requires(T val, size_t lanes) {
    { rotate_left(internal::abi<T>, val, lanes) } -> equivalent_class_as<T>;
};

struct rotate_right_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val, size_t lanes) noexcept {
        lanes %= val.size();
        return dx::slide_right(val, val, lanes);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<E, A> val, size_t lanes) noexcept {
        lanes %= val.size();
        return dx::slide_right(val, val, lanes);
    }

    template <size_t V, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<E, A> val) noexcept {
        return dx::slide_righti<V>(val, val);
    }

public:
    template <fixed_width_class L, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R lanes) noexcept {
        static_assert(R::value > 0);
        constexpr auto V = R::value % simd_abi_traits<L>::size;
        if constexpr (unqualified_rotate_righti<V, L>) {
            if constexpr (canonical_class<L>) {
                if consteval {
                    return fallbacki<V>(arg);
                } else {
                    return rotate_right<V>(internal::abi<L>, arg);
                }
            } else {
                return rotate_right<V>(internal::abi<L>, arg);
            }
        } else if constexpr (canonical_class<L>) {
            return fallbacki<V>(arg);
        } else {
            return operator()(dx::to_canonical(arg), lanes);
        }
    }

    template <scalable_class L, integral_constant_like R>
    requires unqualified_rotate_righti<R::value % simd_abi_traits<L>::size,
                 L> ||
        unqualified_rotate_righti<R::value % simd_abi_traits<L>::size,
            canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R lanes) noexcept {
        static_assert(R::value > 0);
        constexpr auto V = R::value % simd_abi_traits<L>::size;
        if constexpr (unqualified_rotate_righti<V, L>) {
            return rotate_right<V>(internal::abi<L>, arg);
        } else {
            return rotate_right<V>(internal::abi<L>, dx::to_canonical(arg));
        }
    }

    template <fixed_width_class L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t lanes) noexcept {
        if constexpr (unqualified_rotate_right<L>) {
            if constexpr (canonical_class<L>) {
                if consteval {
                    return fallback(arg, lanes);
                } else {
                    return rotate_right(internal::abi<L>, arg, lanes);
                }
            } else {
                return rotate_right(internal::abi<L>, arg, lanes);
            }
        } else if constexpr (canonical_class<L>) {
            return fallback(arg, lanes);
        } else {
            return operator()(dx::to_canonical(arg), lanes);
        }
    }

    template <scalable_class L>
    requires unqualified_rotate_right<L> ||
        unqualified_rotate_right<canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t lanes) noexcept {
        if constexpr (unqualified_rotate_right<L>) {
            return rotate_right(internal::abi<L>, arg, lanes);
        } else {
            return rotate_right(internal::abi<L>, dx::to_canonical(arg), lanes);
        }
    }
};

struct rotate_left_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val, size_t lanes) noexcept {
        lanes %= simd_abi_traits<E, A>::size;
        return dx::slide_left(val, val, lanes);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<E, A> val, size_t lanes) noexcept {
        lanes %= simd_abi_traits<E, A>::size;
        return dx::slide_left(val, val, lanes);
    }

    template <size_t V, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<E, A> val) noexcept {
        return dx::slide_lefti<V>(val, val);
    }

public:
    template <fixed_width_class L, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R lanes) noexcept {
        static_assert(R::value > 0);
        constexpr auto V = R::value % simd_abi_traits<L>::size;
        if constexpr (unqualified_rotate_lefti<V, L>) {
            if constexpr (canonical_class<L>) {
                if consteval {
                    return fallbacki<V>(arg);
                } else {
                    return rotate_left<V>(internal::abi<L>, arg);
                }
            } else {
                return rotate_left<V>(internal::abi<L>, arg);
            }
        } else if constexpr (canonical_class<L>) {
            return fallbacki<V>(arg);
        } else {
            return operator()(dx::to_canonical(arg), lanes);
        }
    }

    template <scalable_class L, integral_constant_like R>
    requires unqualified_rotate_lefti<R::value % simd_abi_traits<L>::size, L> ||
        unqualified_rotate_lefti<R::value % simd_abi_traits<L>::size,
            canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R lanes) noexcept {
        static_assert(R::value > 0);
        constexpr auto V = R::value % simd_abi_traits<L>::size;
        if constexpr (unqualified_rotate_lefti<V, L>) {
            return rotate_left<V>(internal::abi<L>, arg);
        } else {
            return rotate_left<V>(internal::abi<L>, dx::to_canonical(arg));
        }
    }

    template <fixed_width_class L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t lanes) noexcept {
        if constexpr (unqualified_rotate_left<L>) {
            if constexpr (canonical_class<L>) {
                if consteval {
                    return fallback(arg, lanes);
                } else {
                    return rotate_left(internal::abi<L>, arg, lanes);
                }
            } else {
                return rotate_left(internal::abi<L>, arg, lanes);
            }
        } else if constexpr (canonical_class<L>) {
            return fallback(arg, lanes);
        } else {
            return operator()(dx::to_canonical(arg), lanes);
        }
    }

    template <scalable_class L>
    requires unqualified_rotate_left<L> ||
        unqualified_rotate_left<canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t lanes) noexcept {
        if constexpr (unqualified_rotate_left<L>) {
            return rotate_left(internal::abi<L>, arg, lanes);
        } else {
            return rotate_left(internal::abi<L>, dx::to_canonical(arg), lanes);
        }
    }
};

template <size_t V>
struct rotate_righti_t {
public:
    template <simd_class T>
    requires regular_invocable<rotate_right_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return rotate_right_t::operator()(arg, imm<V>);
    }
};

template <size_t V>
struct rotate_lefti_t {
public:
    template <simd_class T>
    requires regular_invocable<rotate_left_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return rotate_left_t::operator()(arg, imm<V>);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <size_t V>
inline constexpr internal::rotate_righti_t<V> rotate_righti{};
DPL_EXPORT inline constexpr internal::rotate_right_t rotate_right{};

DPL_EXPORT template <size_t V>
inline constexpr internal::rotate_lefti_t<V> rotate_lefti{};
DPL_EXPORT inline constexpr internal::rotate_left_t rotate_left{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
