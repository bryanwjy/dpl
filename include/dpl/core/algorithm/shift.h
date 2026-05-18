// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/array_for.h"
#  include "dpl/core/type_traits/common_order_type.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <size_t>
void shift_left(...) noexcept = delete;
template <size_t>
void shift_right(...) noexcept = delete;
void shift_left(...) noexcept = delete;
void shift_right(...) noexcept = delete;

template <size_t V, typename T>
concept unqualified_shift_lefti = requires(T val) {
    { shift_left<V>(internal::abi<T>, val) } -> equivalent_class_as<T>;
};

template <size_t V, typename T>
concept unqualified_shift_righti = requires(T val) {
    { shift_right<V>(internal::abi<T>, val) } -> equivalent_class_as<T>;
};

template <typename T>
concept unqualified_shift_left = requires(T val, size_t shift) {
    { shift_left(internal::abi<T>, val, shift) } -> equivalent_class_as<T>;
};
template <typename T>
concept unqualified_shift_right = requires(T val, size_t shift) {
    { shift_right(internal::abi<T>, val, shift) } -> equivalent_class_as<T>;
};

struct shift_left_t {
private:
    template <typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(
        basic_vector<E, A> val, size_t lanes) noexcept {
        using traits = simd_abi_traits<E, A>;
        lanes = lanes > traits::size ? traits::size : lanes;
        alignas(traits::alignment) E data[2 * traits::size]{};
        dx::store(val, data);
        return dx::load<E, A>(data + lanes);
    }

    template <typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(
        basic_mask<E, A> val, size_t lanes) noexcept {
        return dx::bwshift_left(val, lanes);
    }

    template <size_t V, typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallbacki(basic_vector<E, A> val) noexcept {
        using traits = simd_abi_traits<E, A>;
        if constexpr (V >= traits::size) {
            return dx::broadcast<E, A>(dx::zero);
        } else {
            return [&]<size_t... Is>(index_sequence<Is...>) {
                return dx::initialize<E, A>(
                    (Is + V >= traits::size ? dx::zero : val[imm<Is + V>])...);
            }(make_index_sequence<traits::size>{});
        }
    }

    template <size_t V, typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallbacki(basic_mask<E, A> val) noexcept {
        return dx::bwshift_lefti<V>(val);
    }

public:
    template <fixed_width_class L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t shift) noexcept {
        if constexpr (unqualified_shift_left<L>) {
            if constexpr (canonical_class<L>) {
                if consteval {
                    return fallback(arg, shift);
                } else {
                    return shift_left(internal::abi<L>, arg, shift);
                }
            } else {
                return shift_left(internal::abi<L>, arg, shift);
            }
        } else if constexpr (canonical_class<L>) {
            return fallback(arg, shift);
        } else {
            return operator()(dx::to_canonical(arg), shift);
        }
    }

    template <scalable_class L>
    requires unqualified_shift_left<L> ||
        unqualified_shift_left<canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t shift) noexcept {
        if constexpr (unqualified_shift_left<L>) {
            return shift_left(internal::abi<L>, arg, shift);
        } else {
            return shift_left(internal::abi<L>, dx::to_canonical(arg), shift);
        }
    }

    template <fixed_width_class L, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R shift) noexcept {
        static_assert(R::value > 0);
        if constexpr (unqualified_shift_lefti<R::value, L>) {
            if constexpr (canonical_class<L>) {
                if consteval {
                    return fallbacki<R::value>(arg);
                } else {
                    return shift_left<R::value>(internal::abi<L>, arg);
                }
            } else {
                return shift_left<R::value>(internal::abi<L>, arg);
            }
        } else if constexpr (canonical_class<L>) {
            return fallbacki<R::value>(arg);
        } else {
            return operator()(dx::to_canonical(arg), shift);
        }
    }

    template <scalable_class L, integral_constant_like R>
    requires unqualified_shift_lefti<R::value, L> ||
        unqualified_shift_lefti<R::value, canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R shift) noexcept {
        static_assert(R::value > 0);
        if constexpr (unqualified_shift_lefti<R::value, L>) {
            return shift_left<R::value>(internal::abi<L>, arg);
        } else {
            return shift_left<R::value>(
                internal::abi<L>, dx::to_canonical(arg));
        }
    }
};

struct shift_right_t {
private:
    template <typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val, size_t lanes) noexcept {
        using traits = simd_abi_traits<E, A>;
        lanes = lanes > traits::size ? traits::size : lanes;
        alignas(traits::alignment) E data[2 * traits::size]{};
        dx::store(val, data);
        return dx::load<E, A>(data + lanes);
    }

    template <typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<E, A> val, size_t lanes) noexcept {
        return dx::bwshift_right(val, lanes);
    }

    template <size_t V, typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<E, A> val) noexcept {
        using traits = simd_abi_traits<E, A>;
        if constexpr (V >= traits::size) {
            return dx::broadcast<E, A>(dx::zero);
        } else {
            return [&]<size_t... Is>(index_sequence<Is...>) {
                return dx::initialize<E, A>(
                    (Is < V ? dx::zero : val[imm<Is>])...);
            }(make_index_sequence<traits::size>{});
        }
    }

    template <size_t V, typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_mask<E, A> val) noexcept {
        return dx::bwshift_righti<V>(val);
    }

public:
    template <fixed_width_class L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t shift) noexcept {
        if constexpr (unqualified_shift_right<L>) {
            if constexpr (canonical_class<L>) {
                if consteval {
                    return fallback(arg, shift);
                } else {
                    return shift_right(internal::abi<L>, arg, shift);
                }
            } else {
                return shift_right(internal::abi<L>, arg, shift);
            }
        } else if constexpr (canonical_class<L>) {
            return fallback(arg, shift);
        } else {
            return operator()(dx::to_canonical(arg), shift);
        }
    }

    template <scalable_class L>
    requires unqualified_shift_right<L> ||
        unqualified_shift_right<canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t shift) noexcept {
        if constexpr (unqualified_shift_right<L>) {
            return shift_right(internal::abi<L>, arg, shift);
        } else {
            return shift_right(internal::abi<L>, dx::to_canonical(arg), shift);
        }
    }

    template <fixed_width_class L, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R shift) noexcept {
        static_assert(R::value > 0);
        if constexpr (unqualified_shift_righti<R::value, L>) {
            if constexpr (canonical_class<L>) {
                if consteval {
                    return fallbacki<R::value>(arg);
                } else {
                    return shift_right<R::value>(internal::abi<L>, arg);
                }
            } else {
                return shift_right<R::value>(internal::abi<L>, arg);
            }
        } else if constexpr (canonical_class<L>) {
            return fallbacki<R::value>(arg);
        } else {
            return operator()(dx::to_canonical(arg), shift);
        }
    }

    template <scalable_class L, integral_constant_like R>
    requires unqualified_shift_righti<R::value, L> ||
        unqualified_shift_righti<R::value, canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R shift) noexcept {
        static_assert(R::value > 0);
        if constexpr (unqualified_shift_righti<R::value, L>) {
            return shift_right<R::value>(internal::abi<L>, arg);
        } else {
            return shift_right<R::value>(
                internal::abi<L>, dx::to_canonical(arg));
        }
    }
};

template <size_t V>
struct shift_lefti_t {
public:
    template <simd_class T>
    requires regular_invocable<shift_left_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return shift_left_t::operator()(arg, imm<V>);
    }
};

template <size_t V>
struct shift_righti_t {
public:
    template <simd_class T>
    requires regular_invocable<shift_right_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return shift_right_t::operator()(arg, imm<V>);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <size_t V>
inline constexpr internal::shift_lefti_t<V> shift_lefti{};
DPL_EXPORT template <size_t V>
inline constexpr internal::shift_righti_t<V> shift_righti{};
DPL_EXPORT inline constexpr internal::shift_left_t shift_left{};
DPL_EXPORT inline constexpr internal::shift_right_t shift_right{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
