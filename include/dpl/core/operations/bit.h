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
#  include "dpl/core/concepts/compatible_mask_for.h"
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
void bit_force(...) noexcept = delete;
template <auto>
void bit_force(...) noexcept = delete;
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

struct bit_drop_t : binary_operation_base<bit_drop_t> {
private:
    friend binary_operation_base<bit_drop_t>;

    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { bit_drop(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return bit_drop(internal::abi<A>, left, right);
    }

    template <basic_simd_type M, basic_simd_type T>
    requires same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(M mask, T arg) noexcept {
        return dx::bwandnot(arg, mask);
    }

    template <basic_simd_mask_type M, basic_simd_class T>
    requires same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(M mask, T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::select(mask, arg, dx::all_bits);
        } else {
            return dx::select(mask, arg, dx::broadcast<T>(true));
        }
    }

    template <auto V, basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallbacki(T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::selecti<V>(arg, dx::all_bits);
        } else {
            return dx::selecti<V>(arg, dx::broadcast<T>(true));
        }
    }

public:
    template <basic_simd_type M, basic_simd_type T>
    requires common_bits_simd_with<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_drop(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_drop(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_type M, basic_simd_type T>
    requires common_bits_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<common_bits_simd_t<M, T>> auto {
        using A = common_abi_t<M, T>;
        return bit_drop(internal::abi<A>, mask, arg);
    }

    using binary_operation_base<bit_drop_t>::operator();

    template <basic_simd_mask_type M, basic_simd_type T>
    requires compatible_mask_for<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_drop(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_drop(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_mask_type M, basic_simd_type T>
    requires compatible_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        return bit_drop(internal::abi<A>, mask, arg);
    }

    template <simd_mask_type M, simd_type T>
    requires compatible_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        using A = common_abi_t<M, T>;
        if constexpr (requires { bit_drop(internal::abi<A>, mask, arg); }) {
            return bit_drop(internal::abi<M>, mask, arg);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(arg));
        }
    }

    template <basic_simd_mask_type M, basic_simd_mask_type T>
    requires common_size_simd_with<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_drop(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_drop(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_mask_type M, basic_simd_mask_type T>
    requires common_size_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_size_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        return bit_drop(internal::abi<A>, mask, arg);
    }

    template <simd_mask_type M, simd_mask_type T>
    requires common_size_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        using A = common_abi_t<M, T>;
        if constexpr (requires { bit_drop(internal::abi<A>, mask, arg); }) {
            return bit_drop(internal::abi<M>, mask, arg);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(arg));
        }
    }

    template <basic_simd_class T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_drop<V>(internal::abi<T>, arg); }) {
            if consteval {
                return fallbacki<V>(arg);
            } else {
                return bit_drop<V>(internal::abi<T>, arg);
            }
        } else {
            return fallbacki<V>(arg);
        }
    }

    template <simd_class T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_drop<V>(internal::abi<T>, arg); }) {
            return bit_drop<V>(internal::abi<T>, arg);
        } else {
            return operator()(mask, dx::to_basic_type(arg));
        }
    }
};

template <auto V>
struct bit_dropi_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <simd_class T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<bit_drop_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return bit_drop_t::operator()(mask, arg);
    }
};

struct bit_force_t : binary_operation_base<bit_force_t> {
private:
    friend binary_operation_base<bit_force_t>;

    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { bit_force(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return bit_force(internal::abi<A>, left, right);
    }

    template <basic_simd_type M, basic_simd_type T>
    requires same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(M mask, T arg) noexcept {
        return dx::bwor(mask, arg);
    }

    template <basic_simd_mask_type M, basic_simd_class T>
    requires same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(M mask, T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::select(mask, arg, dx::all_bits);
        } else {
            return dx::select(mask, arg, dx::broadcast<T>(true));
        }
    }

    template <auto V, basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallbacki(T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::selecti<V>(arg, dx::all_bits);
        } else {
            return dx::selecti<V>(arg, dx::broadcast<T>(true));
        }
    }

public:
    template <basic_simd_type M, basic_simd_type T>
    requires common_bits_simd_with<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_force(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_force(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_type M, basic_simd_type T>
    requires common_bits_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<common_bits_simd_t<M, T>> auto {
        using A = common_abi_t<M, T>;
        return bit_force(internal::abi<A>, mask, arg);
    }

    using binary_operation_base<bit_force_t>::operator();

    template <basic_simd_mask_type M, basic_simd_type T>
    requires compatible_mask_for<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_force(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_force(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_mask_type M, basic_simd_type T>
    requires compatible_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        return bit_force(internal::abi<A>, mask, arg);
    }

    template <simd_mask_type M, simd_type T>
    requires compatible_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        if constexpr (requires { bit_force(internal::abi<A>, mask, arg); }) {
            return bit_force(internal::abi<M>, mask, arg);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(arg));
        }
    }

    template <basic_simd_mask_type M, basic_simd_mask_type T>
    requires common_size_simd_with<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_force(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_force(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_mask_type M, basic_simd_mask_type T>
    requires common_size_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_size_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        return bit_force(internal::abi<A>, mask, arg);
    }

    template <simd_mask_type M, simd_mask_type T>
    requires common_size_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_size_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        if constexpr (requires { bit_force(internal::abi<A>, mask, arg); }) {
            return bit_force(internal::abi<M>, mask, arg);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(arg));
        }
    }

    template <basic_simd_class T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_force<V>(internal::abi<T>, arg); }) {
            if consteval {
                return fallbacki<V>(arg);
            } else {
                return bit_force<V>(internal::abi<T>, arg);
            }
        } else {
            return fallbacki<V>(arg);
        }
    }

    template <simd_type T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<T> auto {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_force<V>(internal::abi<T>, arg); }) {
            return bit_force<V>(internal::abi<T>, arg);
        } else {
            return operator()(mask, dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_size_simd_with<T> auto {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_force<V>(internal::abi<T>, arg); }) {
            return bit_force<V>(internal::abi<T>, arg);
        } else {
            return operator()(mask, dx::to_basic_type(arg));
        }
    }
};

template <auto V>
struct bit_forcei_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <simd_class T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<bit_force_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return bit_force_t::operator()(mask, arg);
    }
};

struct bit_keep_t : binary_operation_base<bit_keep_t> {
private:
    friend binary_operation_base<bit_keep_t>;

    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { bit_keep(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return bit_keep(internal::abi<A>, left, right);
    }

    template <basic_simd_type M, basic_simd_type T>
    requires same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(M mask, T arg) noexcept {
        return dx::bwand(mask, arg);
    }

    template <basic_simd_mask_type M, basic_simd_class T>
    requires same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(M mask, T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::select(mask, arg, dx::all_bits);
        } else {
            return dx::select(mask, arg, dx::broadcast<T>(true));
        }
    }

    template <auto V, basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallbacki(T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::selecti<V>(arg, dx::all_bits);
        } else {
            return dx::selecti<V>(arg, dx::broadcast<T>(true));
        }
    }

public:
    template <basic_simd_type M, basic_simd_type T>
    requires common_bits_simd_with<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_keep(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_keep(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_type M, basic_simd_type T>
    requires common_bits_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<common_bits_simd_t<M, T>> auto {
        using A = common_abi_t<M, T>;
        return bit_keep(internal::abi<A>, mask, arg);
    }

    using binary_operation_base<bit_keep_t>::operator();

    template <basic_simd_mask_type M, basic_simd_type T>
    requires compatible_mask_for<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_keep(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_keep(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_mask_type M, basic_simd_type T>
    requires compatible_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        return bit_keep(internal::abi<A>, mask, arg);
    }

    template <simd_mask_type M, simd_type T>
    requires compatible_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        if constexpr (requires { bit_keep(internal::abi<A>, mask, arg); }) {
            return bit_keep(internal::abi<M>, mask, arg);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(arg));
        }
    }

    template <basic_simd_mask_type M, basic_simd_mask_type T>
    requires common_size_simd_with<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_keep(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_keep(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_mask_type M, basic_simd_mask_type T>
    requires common_size_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_size_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        return bit_keep(internal::abi<A>, mask, arg);
    }

    template <simd_mask_type M, simd_mask_type T>
    requires common_size_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_size_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        if constexpr (requires { bit_keep(internal::abi<A>, mask, arg); }) {
            return bit_keep(internal::abi<M>, mask, arg);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(arg));
        }
    }

    template <basic_simd_class T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_keep<V>(internal::abi<T>, arg); }) {
            if consteval {
                return fallbacki<V>(arg);
            } else {
                return bit_keep<V>(internal::abi<T>, arg);
            }
        } else {
            return fallbacki<V>(arg);
        }
    }

    template <simd_type T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<T> auto {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_keep<V>(internal::abi<T>, arg); }) {
            return bit_keep<V>(internal::abi<T>, arg);
        } else {
            return operator()(mask, dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_size_simd_with<T> auto {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_keep<V>(internal::abi<T>, arg); }) {
            return bit_keep<V>(internal::abi<T>, arg);
        } else {
            return operator()(mask, dx::to_basic_type(arg));
        }
    }
};

template <auto V>
struct bit_keepi_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <simd_class T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<bit_keep_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return bit_keep_t::operator()(mask, arg);
    }
};

struct bit_stencil_t : binary_operation_base<bit_stencil_t> {
private:
    friend binary_operation_base<bit_stencil_t>;

    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { bit_stencil(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return bit_stencil(internal::abi<A>, left, right);
    }

    template <basic_simd_type M, basic_simd_type T>
    requires same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(M mask, T arg) noexcept {
        return dx::bwornot(arg, mask);
    }

    template <basic_simd_mask_type M, basic_simd_class T>
    requires same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(M mask, T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::select(mask, arg, dx::all_bits);
        } else {
            return dx::select(mask, arg, dx::broadcast<T>(true));
        }
    }

    template <auto V, basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallbacki(T arg) noexcept {
        if constexpr (basic_simd_type<T>) {
            return dx::selecti<V>(arg, dx::all_bits);
        } else {
            return dx::selecti<V>(arg, dx::broadcast<T>(true));
        }
    }

public:
    template <basic_simd_type M, basic_simd_type T>
    requires common_bits_simd_with<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_stencil(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_stencil(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_type M, basic_simd_type T>
    requires common_bits_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<common_bits_simd_t<M, T>> auto {
        using A = common_abi_t<M, T>;
        return bit_stencil(internal::abi<A>, mask, arg);
    }

    using binary_operation_base<bit_stencil_t>::operator();

    template <basic_simd_mask_type M, basic_simd_type T>
    requires compatible_mask_for<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_stencil(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_stencil(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_mask_type M, basic_simd_type T>
    requires compatible_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        return bit_stencil(internal::abi<A>, mask, arg);
    }

    template <simd_mask_type M, simd_type T>
    requires compatible_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        if constexpr (requires { bit_stencil(internal::abi<A>, mask, arg); }) {
            return bit_stencil(internal::abi<M>, mask, arg);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(arg));
        }
    }

    template <basic_simd_mask_type M, basic_simd_mask_type T>
    requires common_size_simd_with<M, T> && same_abi_simd_as<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (requires { bit_stencil(internal::abi<M>, mask, arg); }) {
            if consteval {
                return fallback(mask, arg);
            } else {
                return bit_stencil(internal::abi<M>, mask, arg);
            }
        } else {
            return fallback(mask, arg);
        }
    }

    template <basic_simd_mask_type M, basic_simd_mask_type T>
    requires common_size_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_size_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        return bit_stencil(internal::abi<A>, mask, arg);
    }

    template <simd_mask_type M, simd_mask_type T>
    requires common_size_simd_with<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_size_simd_with<T> auto {
        using A = common_abi_t<M, T>;
        if constexpr (requires { bit_stencil(internal::abi<A>, mask, arg); }) {
            return bit_stencil(internal::abi<M>, mask, arg);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(arg));
        }
    }

    template <basic_simd_class T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_stencil<V>(internal::abi<T>, arg); }) {
            if consteval {
                return fallbacki<V>(arg);
            } else {
                return bit_stencil<V>(internal::abi<T>, arg);
            }
        } else {
            return fallbacki<V>(arg);
        }
    }

    template <simd_type T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_bits_simd_with<T> auto {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_stencil<V>(internal::abi<T>, arg); }) {
            return bit_stencil<V>(internal::abi<T>, arg);
        } else {
            return operator()(mask, dx::to_basic_type(arg));
        }
    }

    template <simd_mask_type T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> common_size_simd_with<T> auto {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { bit_stencil<V>(internal::abi<T>, arg); }) {
            return bit_stencil<V>(internal::abi<T>, arg);
        } else {
            return operator()(mask, dx::to_basic_type(arg));
        }
    }
};

template <auto V>
struct bit_stencili_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <simd_class T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<bit_stencil_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return bit_stencil_t::operator()(mask, arg);
    }
};

struct bit_select_t {
private:
    template <auto V, basic_simd_element ET, basic_simd_element EF, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallbacki(
        basic_simd<ET, A> tval, basic_simd<EF, A> fval) noexcept {
        using ER = common_bits_type_t<ET, EF>;
        return dx::selecti<V>(
            dx::reinterpret<ER>(tval), dx::reinterpret<ER>(fval));
    }

    template <auto V, basic_simd_element ET, basic_simd_element EF, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallbacki(
        simd_mask<ET, A> tval, simd_mask<EF, A> fval) noexcept {
        using ER = common_size_type_t<ET, EF>;
        return dx::selecti<V>(
            dx::reinterpret<ER>(tval), dx::reinterpret<ER>(fval));
    }

    template <basic_simd_element EM, basic_simd_element ET,
        basic_simd_element EF, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(simd_mask<EM, A> mask,
        basic_simd<ET, A> tval, basic_simd<EF, A> fval) noexcept {
        using ER = common_bits_type_t<ET, EF>;
        return dx::select(
            mask, dx::reinterpret<ER>(tval), dx::reinterpret<ER>(fval));
    }

    template <basic_simd_element EM, basic_simd_element ET,
        basic_simd_element EF, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(simd_mask<EM, A> mask, simd_mask<ET, A> tval,
        simd_mask<EF, A> fval) noexcept {
        using ER = common_size_type_t<ET, EF>;
        return dx::select(
            mask, dx::reinterpret<ER>(tval), dx::reinterpret<ER>(fval));
    }

    template <basic_simd_element EM, basic_simd_element ET,
        basic_simd_element EF, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd<EM, A> mask,
        basic_simd<ET, A> tval, basic_simd<EF, A> fval) noexcept {
        using ER = common_bits_type_t<ET, EF, EM>;
        return dx::bwor(
            dx::bwand(dx::reinterpret<ER>(mask), dx::reinterpret<ER>(tval)),
            dx::bwandnot(dx::reinterpret<ER>(fval), dx::reinterpret<ER>(mask)));
    }

    template <basic_simd_class T, basic_simd_class F, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto impl(M mask, T tval, F fval) {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires {
                          bit_select<V>(internal::abi<T>, tval, fval);
                      }) {
            if consteval {
                return fallbacki<V>(tval, fval);
            } else {
                return bit_select<V>(internal::abi<T>, tval, fval);
            }
        } else {
            return fallbacki<V>(tval, fval);
        }
    }

    template <basic_simd_class M, basic_simd_class T, basic_simd_class F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto impl(M mask, T tval, F fval) {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires {
                          bit_select(internal::abi<T>, mask, tval, fval);
                      }) {
            if consteval {
                return fallback(mask, tval, fval);
            } else {
                return bit_select(internal::abi<T>, mask, tval, fval);
            }
        } else {
            return fallback(mask, tval, fval);
        }
    }

    template <simd_class T, simd_class F, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto impl(M mask, T tval, F fval) {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires {
                          bit_select<V>(internal::abi<T>, tval, fval);
                      }) {
            return bit_select<V>(internal::abi<T>, tval, fval);
        } else {
            return operator()(
                mask, dx::to_basic_type(tval), dx::to_basic_type(fval));
        }
    }

    template <simd_class M, simd_class T, simd_class F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto impl(M mask, T tval, F fval) {
        using A = common_abi_t<T, F, M>;
        if constexpr (requires {
                          bit_select(internal::abi<A>, mask, tval, fval);
                      }) {
            return bit_select(internal::abi<A>, mask, tval, fval);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(tval),
                dx::to_basic_type(fval));
        }
    }

public:
    template <basic_simd_type T, basic_simd_type F, immediate_mask_for<T> M>
    requires common_bits_simd_with<T, F> && same_abi_simd_as<T, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept {
        return impl(mask, tval, fval);
    }

    template <basic_simd_type T, basic_simd_type F, immediate_mask_for<T> M>
    requires common_bits_simd_with<T, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept
        -> common_bits_simd_with<common_bits_simd_t<T, F>> auto {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        using A = common_abi_t<T, F>;
        return bit_select<V>(internal::abi<A>, tval, fval);
    }

    template <basic_simd_mask_type T, basic_simd_mask_type F,
        immediate_mask_for<T> M>
    requires common_size_simd_with<T, F> && same_abi_simd_as<T, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept {
        return impl(mask, tval, fval);
    }

    template <basic_simd_mask_type T, basic_simd_mask_type F,
        immediate_mask_for<T> M>
    requires common_size_simd_with<T, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept
        -> common_size_simd_with<common_size_simd_t<T, F>> auto {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        using A = common_abi_t<T, F>;
        return bit_select<V>(internal::abi<A>, tval, fval);
    }

    template <simd_type T, simd_type F, immediate_mask_for<T> M>
    requires common_bits_simd_with<T, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept {
        return impl(mask, tval, fval);
    }

    template <simd_mask_type T, simd_mask_type F, immediate_mask_for<T> M>
    requires common_size_simd_with<T, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept {
        return impl(mask, tval, fval);
    }

    template <basic_simd_mask_type M, basic_simd_type T, basic_simd_type F>
    requires common_bits_simd_with<T, F> &&
        common_size_simd_with<common_bits_simd_t<T, F>, M> &&
        same_abi_simd_as<T, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept {
        return impl(mask, tval, fval);
    }

    template <basic_simd_mask_type M, basic_simd_type T, basic_simd_type F>
    requires common_bits_simd_with<T, F> &&
        common_size_simd_with<common_bits_simd_t<T, F>, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept
        -> common_bits_simd_with<
            common_size_simd_t<common_bits_simd_t<T, F>, M>> auto {
        using A = common_abi_t<T, F, M>;
        return bit_select(internal::abi<A>, mask, tval, fval);
    }

    template <basic_simd_mask_type M, basic_simd_mask_type T,
        basic_simd_mask_type F>
    requires common_size_simd_with<T, F> &&
        common_size_simd_with<common_size_simd_t<T, F>, M> &&
        same_abi_simd_as<T, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept {
        return impl(mask, tval, fval);
    }

    template <basic_simd_mask_type M, basic_simd_mask_type T,
        basic_simd_mask_type F>
    requires common_size_simd_with<T, F> &&
        common_size_simd_with<common_size_simd_t<T, F>, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept
        -> common_size_simd_with<common_size_simd_t<T, F, M>> auto {
        using A = common_abi_t<T, F, M>;
        return bit_select(internal::abi<A>, mask, tval, fval);
    }

    template <simd_mask_type M, simd_type T, simd_type F>
    requires common_bits_simd_with<T, F> &&
        common_size_simd_with<common_bits_simd_t<T, F>, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept {
        return impl(mask, tval, fval);
    }

    template <simd_mask_type M, simd_mask_type T, simd_mask_type F>
    requires common_size_simd_with<T, F> &&
        common_size_simd_with<common_size_simd_t<T, F>, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T tval, F fval) noexcept {
        return impl(mask, tval, fval);
    }
};

template <auto V>
struct bit_selecti_t : binary_operation_base<bit_selecti_t<V>> {
private:
    friend binary_operation_base<bit_selecti_t>;

    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

    template <simd_abi A, typename L, typename R>
    requires requires(
        L lhs, R rhs) { bit_select<V>(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return bit_select<V>(internal::abi<A>, left, right);
    }

public:
    template <simd_class T, simd_class F>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<bit_select_t, mask_type<T>, T, F>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T tval, F fval) noexcept {
        constexpr mask_type<T> mask{};
        return bit_select_t::operator()(mask, tval, fval);
    }

    using binary_operation_base<bit_selecti_t>::operator();
};

struct popcount_t {
private:
    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd<E, A> arg) noexcept {
        return internal::transform<basic_simd<E, A>>(arg, [](auto val) {
            auto const count = __DPL popcount(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr int fallback(simd_mask<E, A> arg) noexcept {
        return []<size_t... Is>(simd_mask<E, A> arg, index_sequence<Is...>) {
            return (... + static_cast<int>(arg[Is]));
        }(arg, iota_sequence<E, A>);
    }

public:
    template <basic_simd_type T>
    requires integral_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { popcount(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return popcount(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <integral_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { popcount(internal::abi<T>, arg); }) {
            return popcount(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <basic_simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { popcount(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return popcount(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { popcount(internal::abi<T>, arg); }) {
            return popcount(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T) noexcept {
        return __DPL popcount(__DPL to_unsigned(T::value));
    }
};

struct countl_zero_t {
private:
    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd<E, A> arg) noexcept {
        return internal::transform<basic_simd<E, A>>(arg, [](auto val) {
            auto const count = __DPL countl_zero(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(simd_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, simd_mask<E, A> arg,
                   immediate<I> idx) -> int {
            auto const val = static_cast<int>(!arg[idx]);
            if constexpr (I == 0) {
                return val;
            } else {
                return val + (-val & self(arg, imm<I - 1>));
            }
        }(arg, imm<element_count<E, A> - 1>);
    }

public:
    template <basic_simd_type T>
    requires integral_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countl_zero(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return countl_zero(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <integral_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countl_zero(internal::abi<T>, arg); }) {
            return countl_zero(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <basic_simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countl_zero(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return countl_zero(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countl_zero(internal::abi<T>, arg); }) {
            return countl_zero(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T) noexcept {
        return __DPL countl_zero(__DPL to_unsigned(T::value));
    }
};

struct countl_one_t {
private:
    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd<E, A> arg) noexcept {
        return internal::transform<basic_simd<E, A>>(arg, [](auto val) {
            auto const count = __DPL countl_one(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(simd_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, simd_mask<E, A> arg,
                   immediate<I> idx) -> int {
            auto const val = static_cast<int>(arg[idx]);
            if constexpr (I == 0) {
                return val;
            } else {
                return val + (-val & self(arg, imm<I - 1>));
            }
        }(arg, imm<element_count<E, A> - 1>);
    }

public:
    template <basic_simd_type T>
    requires integral_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countl_one(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return countl_one(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <integral_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countl_one(internal::abi<T>, arg); }) {
            return countl_one(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <basic_simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countl_one(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return countl_one(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countl_one(internal::abi<T>, arg); }) {
            return countl_one(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T) noexcept {
        return __DPL countl_one(__DPL to_unsigned(T::value));
    }
};

struct countr_zero_t {
private:
    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd<E, A> arg) noexcept {
        return internal::transform<basic_simd<E, A>>(arg, [](auto val) {
            auto const count = __DPL countr_zero(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(simd_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, simd_mask<E, A> arg,
                   immediate<I> idx) -> int {
            auto const val = static_cast<int>(!arg[idx]);
            if constexpr (I == element_count<E, A>) {
                return 0;
            } else {
                return val + (-val & self(arg, imm<I + 1>));
            }
        }(arg, imm<0>);
    }

public:
    template <basic_simd_type T>
    requires integral_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countr_zero(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return countr_zero(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <integral_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countr_zero(internal::abi<T>, arg); }) {
            return countr_zero(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <basic_simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countr_zero(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return countr_zero(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countr_zero(internal::abi<T>, arg); }) {
            return countr_zero(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T) noexcept {
        return __DPL countr_zero(__DPL to_unsigned(T::value));
    }
};

struct countr_one_t {
private:
    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd<E, A> arg) noexcept {
        return internal::transform<basic_simd<E, A>>(arg, [](auto val) {
            auto const count = __DPL countr_one(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(simd_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, simd_mask<E, A> arg,
                   immediate<I> idx) -> int {
            auto const val = static_cast<int>(arg[idx]);
            if constexpr (I == element_count<E, A>) {
                return 0;
            } else {
                return val + (-val & self(arg, imm<I + 1>));
            }
        }(arg, imm<0>);
    }

public:
    template <basic_simd_type T>
    requires integral_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countr_one(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return countr_one(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <integral_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countr_one(internal::abi<T>, arg); }) {
            return countr_one(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <basic_simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countr_one(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return countr_one(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { countr_one(internal::abi<T>, arg); }) {
            return countr_one(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr int DPL_VECTORCALL operator()(T) noexcept {
        return __DPL countr_one(__DPL to_unsigned(T::value));
    }
};

struct byteswap_t {
private:
    template <integral E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_simd<E, A> arg) noexcept {
        return internal::transform<basic_simd<E, A>>(
            arg, [](auto val) { return __DPL byteswap(val); });
    }

public:
    template <basic_simd_type T>
    requires integral_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires { byteswap(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return byteswap(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <integral_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept
        -> equivalent_simd_as<T> auto {
        if constexpr (requires { byteswap(internal::abi<T>, arg); }) {
            return byteswap(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <integral_constant_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T) noexcept {
        return __DPL byteswap(T::value);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bit_drop_t bit_drop{};
DPL_EXPORT inline constexpr internal::bit_force_t bit_force{};
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
inline constexpr internal::bit_forcei_t<V> bit_forcei{};
DPL_EXPORT template <auto V>
inline constexpr internal::bit_keepi_t<V> bit_keepi{};
DPL_EXPORT template <auto V>
inline constexpr internal::bit_stencili_t<V> bit_stencili{};
DPL_EXPORT template <auto V>
inline constexpr internal::bit_selecti_t<V> bit_selecti{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
