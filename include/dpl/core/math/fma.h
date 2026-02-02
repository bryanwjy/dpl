// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/negate.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename T>
struct fma_base {
    template <basic_simd_type TA, broadcastable_to<TA> TB,
        broadcastable_to<TA> TC>
    requires regular_invocable<T, TA, TA, TA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, TA, TA, TB, TC>) {
            if consteval {
                return T::operator()(
                    a, dx::broadcast<TA>(b), dx::broadcast<TA>(c));
            } else {
                return T::native(internal::abi<TA>, a, b, c);
            }
        } else {
            return T::operator()(a, dx::broadcast<TA>(b), dx::broadcast<TA>(c));
        }
    }

    template <basic_simd_type TB, broadcastable_to<TB> TA,
        broadcastable_to<TB> TC>
    requires regular_invocable<T, TB, TB, TB>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, TB, TA, TB, TC>) {
            if consteval {
                return T::operator()(
                    dx::broadcast<TB>(a), b, dx::broadcast<TB>(c));
            } else {
                return T::native(internal::abi<TA>, a, b, c);
            }
        } else {
            return T::operator()(dx::broadcast<TB>(a), b, dx::broadcast<TB>(c));
        }
    }

    template <basic_simd_type TC, broadcastable_to<TC> TA,
        broadcastable_to<TC> TB>
    requires regular_invocable<T, TC, TC, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, TC, TA, TB, TC>) {
            if consteval {
                return T::operator()(
                    dx::broadcast<TC>(a), dx::broadcast<TC>(b), c);
            } else {
                return T::native(internal::abi<TA>, a, b, c);
            }
        } else {
            return T::operator()(dx::broadcast<TC>(a), dx::broadcast<TC>(b), c);
        }
    }

    template <basic_simd_type TA, broadcastable_to<TA> TC>
    requires regular_invocable<T, TA, TA, TA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TA b, TC c) noexcept {
        if constexpr (implements_native<T, TA, TA, TA, TC>) {
            if consteval {
                return T::operator()(a, b, dx::broadcast<TA>(c));
            } else {
                return T::native(internal::abi<TA>, a, b, c);
            }
        } else {
            return T::operator()(a, b, dx::broadcast<TA>(c));
        }
    }

    template <basic_simd_type TA, broadcastable_to<TA> TB>
    requires regular_invocable<T, TA, TA, TA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TA c) noexcept {
        if constexpr (implements_native<T, TA, TA, TB, TA>) {
            if consteval {
                return T::operator()(a, dx::broadcast<TA>(b), c);
            } else {
                return T::native(internal::abi<TA>, a, b, c);
            }
        } else {
            return T::operator()(a, dx::broadcast<TA>(b), c);
        }
    }

    template <basic_simd_type TB, broadcastable_to<TB> TA>
    requires regular_invocable<T, TB, TB, TB>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TB c) noexcept {
        if constexpr (implements_native<T, TA, TA, TB, TB>) {
            if consteval {
                return T::operator()(dx::broadcast<TB>(a), b, c);
            } else {
                return T::native(internal::abi<TA>, a, b, c);
            }
        } else {
            return T::operator()(dx::broadcast<TB>(a), b, c);
        }
    }

    template <simd_type TA, common_arithmetic_simd_with<TA> TB,
        broadcastable_to<common_arithmetic_simd_t<TA, TB>> TC>
    requires regular_invocable<T, TA, TB, common_arithmetic_simd_t<TA, TB>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, common_abi_t<TA, TB>, TA, TB, TC>) {
            return T::native(internal::abi<common_abi_t<TA, TB>>, a, b, c);
        } else {
            return T::operator()(
                a, b, dx::broadcast<common_arithmetic_simd_t<TA, TB>>(c));
        }
    }

    template <simd_type TA, common_arithmetic_simd_with<TA> TC,
        broadcastable_to<common_arithmetic_simd_t<TA, TC>> TB>
    requires regular_invocable<T, TA, common_arithmetic_simd_t<TA, TC>, TB>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, common_abi_t<TA, TC>, TA, TB, TC>) {
            return T::native(internal::abi<common_abi_t<TA, TC>>, a, b, c);
        } else {
            return T::operator()(
                a, dx::broadcast<common_arithmetic_simd_t<TA, TC>>(b), c);
        }
    }

    template <simd_type TB, common_arithmetic_simd_with<TB> TC,
        broadcastable_to<common_arithmetic_simd_t<TB, TC>> TA>
    requires regular_invocable<T, common_arithmetic_simd_t<TB, TC>, TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, common_abi_t<TA, TC>, TA, TB, TC>) {
            return T::native(internal::abi<common_abi_t<TA, TC>>, a, b, c);
        } else {
            return T::operator()(
                dx::broadcast<common_arithmetic_simd_t<TB, TC>>(a), b, c);
        }
    }

    template <simd_type TA, broadcastable_to<TA> TB, broadcastable_to<TA> TC>
    requires regular_invocable<T, TA, TA, TA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, TA, TA, TB, TC>) {
            return T::native(internal::abi<TA>, a, b, c);
        } else {
            return T::operator()(a, dx::broadcast<TA>(b), dx::broadcast<TA>(c));
        }
    }

    template <simd_type TB, broadcastable_to<TB> TA, broadcastable_to<TB> TC>
    requires regular_invocable<T, TB, TB, TB>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, TB, TA, TB, TC>) {
            return T::native(internal::abi<TA>, a, b, c);
        } else {
            return T::operator()(dx::broadcast<TB>(a), b, dx::broadcast<TB>(c));
        }
    }

    template <simd_type TC, broadcastable_to<TC> TA, broadcastable_to<TC> TB>
    requires regular_invocable<T, TC, TC, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, TC, TA, TB, TC>) {
            return T::native(internal::abi<TA>, a, b, c);
        } else {
            return T::operator()(dx::broadcast<TC>(a), dx::broadcast<TC>(b), c);
        }
    }
};

struct fmadd_t : fma_base<fmadd_t> {
private:
    friend fma_base<fmadd_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    requires requires(TA a, TB b, TC c) { fmadd(internal::abi<A>, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(TA a, TB b, TC c) noexcept {
        return fmadd(internal::abi<A>, a, b, c);
    }

    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(
        basic_simd<E, A> a, basic_simd<E, A> b, basic_simd<E, A> c) noexcept {
        return dx::add(dx::mul(a, b), c);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T a, T b, T c) noexcept {
        if constexpr (requires { fmadd(internal::abi<T>, a, b, c); }) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return fmadd(internal::abi<T>, a, b, c);
            }
        } else {
            return fallback(a, b, c);
        }
    }

    template <basic_simd_type A, basic_simd_type B, basic_simd_type C>
    requires common_arithmetic_simd_with<A, B> &&
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>, C> &&
        floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        return fmadd(internal::abi<common_abi_t<A, B, C>>, a, b, c);
    }

    template <simd_type A, common_arithmetic_simd_with<A> B,
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>> C>
    requires floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        if constexpr (requires {
                          fmadd(internal::abi<common_abi_t<A, B, C>>, a, b, c);
                      }) {
            return fmadd(internal::abi<common_abi_t<A, B, C>>, a, b, c);
        } else {
            return operator()(dx::to_basic_type(a), dx::to_basic_type(b),
                dx::to_basic_type(c));
        }
    }

    using fma_base<fmadd_t>::operator();
};

struct fmsub_t : fma_base<fmsub_t> {
private:
    friend fma_base<fmsub_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    requires requires(TA a, TB b, TC c) { fmsub(internal::abi<A>, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(TA a, TB b, TC c) noexcept {
        return fmsub(internal::abi<A>, a, b, c);
    }

    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(
        basic_simd<E, A> a, basic_simd<E, A> b, basic_simd<E, A> c) noexcept {
        return dx::sub(dx::mul(a, b), c);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T a, T b, T c) noexcept {
        if constexpr (requires { fmsub(internal::abi<T>, a, b, c); }) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return fmsub(internal::abi<T>, a, b, c);
            }
        } else {
            return fallback(a, b, c);
        }
    }

    template <basic_simd_type A, basic_simd_type B, basic_simd_type C>
    requires common_arithmetic_simd_with<A, B> &&
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>, C> &&
        floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        return fmsub(internal::abi<common_abi_t<A, B, C>>, a, b, c);
    }

    template <simd_type A, common_arithmetic_simd_with<A> B,
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>> C>
    requires floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        if constexpr (requires {
                          fmsub(internal::abi<common_abi_t<A, B, C>>, a, b, c);
                      }) {
            return fmsub(internal::abi<common_abi_t<A, B, C>>, a, b, c);
        } else {
            return operator()(dx::to_basic_type(a), dx::to_basic_type(b),
                dx::to_basic_type(c));
        }
    }

    using fma_base<fmsub_t>::operator();
};

struct fnmadd_t : fma_base<fnmadd_t> {
private:
    friend fma_base<fnmadd_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    requires requires(TA a, TB b, TC c) { fnmadd(internal::abi<A>, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(TA a, TB b, TC c) noexcept {
        return fnmadd(internal::abi<A>, a, b, c);
    }

    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(
        basic_simd<E, A> a, basic_simd<E, A> b, basic_simd<E, A> c) noexcept {
        return dx::sub(c, dx::mul(a, b));
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T a, T b, T c) noexcept {
        if constexpr (requires { fnmadd(internal::abi<T>, a, b, c); }) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return fnmadd(internal::abi<T>, a, b, c);
            }
        } else {
            return fallback(a, b, c);
        }
    }

    template <basic_simd_type A, basic_simd_type B, basic_simd_type C>
    requires common_arithmetic_simd_with<A, B> &&
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>, C> &&
        floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        return fnmadd(internal::abi<common_abi_t<A, B, C>>, a, b, c);
    }

    template <simd_type A, common_arithmetic_simd_with<A> B,
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>> C>
    requires floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        if constexpr (requires {
                          fnmadd(internal::abi<common_abi_t<A, B, C>>, a, b, c);
                      }) {
            return fnmadd(internal::abi<common_abi_t<A, B, C>>, a, b, c);
        } else {
            return operator()(dx::to_basic_type(a), dx::to_basic_type(b),
                dx::to_basic_type(c));
        }
    }

    using fma_base<fnmadd_t>::operator();
};

struct fnmsub_t : fma_base<fnmsub_t> {
private:
    friend fma_base<fnmsub_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    requires requires(TA a, TB b, TC c) { fnmsub(internal::abi<A>, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(TA a, TB b, TC c) noexcept {
        return fnmsub(internal::abi<A>, a, b, c);
    }

    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(
        basic_simd<E, A> a, basic_simd<E, A> b, basic_simd<E, A> c) noexcept {
        return dx::sub(dx::negate(dx::mul(a, b)), c);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T a, T b, T c) noexcept {
        if constexpr (requires { fnmsub(internal::abi<T>, a, b, c); }) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return fnmsub(internal::abi<T>, a, b, c);
            }
        } else {
            return fallback(a, b, c);
        }
    }

    template <basic_simd_type A, basic_simd_type B, basic_simd_type C>
    requires common_arithmetic_simd_with<A, B> &&
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>, C> &&
        floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        return fnmsub(internal::abi<common_abi_t<A, B, C>>, a, b, c);
    }

    template <simd_type A, common_arithmetic_simd_with<A> B,
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>> C>
    requires floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        if constexpr (requires {
                          fnmsub(internal::abi<common_abi_t<A, B, C>>, a, b, c);
                      }) {
            return fnmsub(internal::abi<common_abi_t<A, B, C>>, a, b, c);
        } else {
            return operator()(dx::to_basic_type(a), dx::to_basic_type(b),
                dx::to_basic_type(c));
        }
    }

    using fma_base<fnmsub_t>::operator();
};

struct fmaddsub_t : fma_base<fmaddsub_t> {
private:
    friend fma_base<fmaddsub_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    requires requires(TA a, TB b, TC c) { fmaddsub(internal::abi<A>, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(TA a, TB b, TC c) noexcept {
        return fmaddsub(internal::abi<A>, a, b, c);
    }

    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(
        basic_simd<E, A> a, basic_simd<E, A> b, basic_simd<E, A> c) noexcept {
        return dx::selecti<0b1010>(
            fmadd_t::operator()(a, b, c), fmsub_t::operator()(a, b, c));
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T a, T b, T c) noexcept {
        if constexpr (requires { fmaddsub(internal::abi<T>, a, b, c); }) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return fmaddsub(internal::abi<T>, a, b, c);
            }
        } else {
            return fallback(a, b, c);
        }
    }

    template <basic_simd_type A, basic_simd_type B, basic_simd_type C>
    requires common_arithmetic_simd_with<A, B> &&
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>, C> &&
        floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        return fmaddsub(internal::abi<common_abi_t<A, B, C>>, a, b, c);
    }

    template <simd_type A, common_arithmetic_simd_with<A> B,
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>> C>
    requires floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        if constexpr (requires {
                          fmaddsub(
                              internal::abi<common_abi_t<A, B, C>>, a, b, c);
                      }) {
            return fmaddsub(internal::abi<common_abi_t<A, B, C>>, a, b, c);
        } else {
            return operator()(dx::to_basic_type(a), dx::to_basic_type(b),
                dx::to_basic_type(c));
        }
    }

    using fma_base<fmaddsub_t>::operator();
};

struct fmsubadd_t : fma_base<fmsubadd_t> {
private:
    friend fma_base<fmsubadd_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    requires requires(TA a, TB b, TC c) { fmsubadd(internal::abi<A>, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(TA a, TB b, TC c) noexcept {
        return fmsubadd(internal::abi<A>, a, b, c);
    }

    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(
        basic_simd<E, A> a, basic_simd<E, A> b, basic_simd<E, A> c) noexcept {
        return dx::selecti<0b1010>(
            fmsub_t::operator()(a, b, c), fmadd_t::operator()(a, b, c));
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T a, T b, T c) noexcept {
        if constexpr (requires { fmsubadd(internal::abi<T>, a, b, c); }) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return fmsubadd(internal::abi<T>, a, b, c);
            }
        } else {
            return fallback(a, b, c);
        }
    }

    template <basic_simd_type A, basic_simd_type B, basic_simd_type C>
    requires common_arithmetic_simd_with<A, B> &&
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>, C> &&
        floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        return fmsubadd(internal::abi<common_abi_t<A, B, C>>, a, b, c);
    }

    template <simd_type A, common_arithmetic_simd_with<A> B,
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>> C>
    requires floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        if constexpr (requires {
                          fmsubadd(
                              internal::abi<common_abi_t<A, B, C>>, a, b, c);
                      }) {
            return fmsubadd(internal::abi<common_abi_t<A, B, C>>, a, b, c);
        } else {
            return operator()(dx::to_basic_type(a), dx::to_basic_type(b),
                dx::to_basic_type(c));
        }
    }

    using fma_base<fmsubadd_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::fmadd_t fmadd{};
DPL_EXPORT inline constexpr internal::fmsub_t fmsub{};
DPL_EXPORT inline constexpr internal::fnmadd_t fnmadd{};
DPL_EXPORT inline constexpr internal::fnmsub_t fnmsub{};
DPL_EXPORT inline constexpr internal::fmaddsub_t fmaddsub{};
DPL_EXPORT inline constexpr internal::fmsubadd_t fmsubadd{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
