// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fma.h"
#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/constants/digits.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/type_traits/remove_const.h"
#  include "dpl/std/utility/forward.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace dx = __DPL datapar;
template <floating_point auto... Vs>
struct polynomial {
    static_assert(sizeof...(Vs) >= 2);
    template <floating_point T>
    struct coeffs_t {
        template <size_t I>
        static consteval T operator[](immediate<I>) noexcept {
#if __cpp_pack_indexing >= 202311L & (DPL_CXX26 | DPL_COMPILER_CLANG)
#  if DPL_COMPILER_CLANG & !DPL_CXX26
            DPL_DISABLE_WARNING_PUSH()
            DPL_DISABLE_WARNING("-Wc++26-extensions")
#  endif
            static_assert(
                digits_v<T> <= digits_v<remove_const_t<decltype(Vs...[I])>>);
            return static_cast<T>(Vs...[I]);
#  if DPL_COMPILER_CLANG & !DPL_CXX26
            DPL_DISABLE_WARNING_POP()
#  endif
#else
            static_assert((... &&
                (digits_v<T> <= digits_v<remove_const_t<decltype(Vs)>>)));
            using array = T[sizeof...(Vs)];
            return array{Vs...}[I];
#endif
        }

        consteval T back(this coeffs_t self) noexcept {
            return self[imm<sizeof...(Vs) - 1>];
        }
    };

    template <floating_point T>
    static constexpr coeffs_t<T> coeffs;

    static consteval auto front(auto val, auto...) noexcept { return val; }

private:
    template <basic_simd_type T0, same_abi_simd_as<T0>... Ts>
    requires floating_point_simd<T0> &&
        (... && (floating_point_simd<Ts> && basic_simd_type<Ts>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        estrin(T0 x, T0 t0, Ts... ts) noexcept {
        using simd = T0;
        constexpr size_t size = (sizeof...(Ts) + 1);
        static constexpr auto terms = []<size_t I>(this auto self, immediate<I>,
                                          auto&& first,
                                          auto&&... params) -> decltype(auto) {
            if constexpr (I == 0) {
                return first;
            } else {
                return self(
                    imm<I - 1>, __DPL forward<decltype(params)>(params)...);
            }
        };

        auto x2 = x * x;
        auto accumulate = [&]<size_t I = 1>(
                              this auto self, simd xi, immediate<I> = {}) {
            static_assert(I >= 1);
            t0 = dx::fmadd(terms(imm<I>, t0, ts...), xi, t0);
            if constexpr (I + 1 < size) {
                self(xi * x2, imm<I + 1>);
            }
        };

        if constexpr ((sizeof...(Vs) % 2) == 0) {
            accumulate(x2);
        } else {
            accumulate(x);
        }

        return t0;
    }

    template <floating_point T, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL estrin(basic_simd<T, A> x) noexcept {
        // Can define more if needed, but for now, limit to 8
        static_assert(sizeof...(Vs) <= 8);
        if constexpr (sizeof...(Vs) == 8) {
            static constexpr size_t size = sizeof...(Vs) / 2;
            static constexpr make_index_sequence<size> indices{};
            return []<size_t... Is>(index_sequence<Is...>, auto x) {
                constexpr polynomial<coeffs<T>[imm<Is>]...> first{};
                constexpr polynomial<coeffs<T>[imm<size + Is>]...> second{};
                auto x2 = x * x;
                return dx::fmadd(x2 * x2, first(x), second(x));
            }(indices, x);
        } else {
            using simd = basic_simd<T, A>;
            constexpr size_t size = (sizeof...(Vs) + 1) / 2;
            constexpr make_index_sequence<size> indices{};
            return []<size_t... Is>(index_sequence<Is...>, auto x) {
                if constexpr ((sizeof...(Vs) % 2) == 0) {
                    return estrin(x,
                        dx::fmadd(coeffs<T>[imm<Is * 2 + 1>], x,
                            coeffs<T>[imm<Is * 2>])...);
                } else {
                    return estrin(x,
                        dx::fmadd(coeffs<T>[imm<Is * 2 + 1>], x,
                            coeffs<T>[imm<Is * 2>])...,
                        dx::broadcast<A>(coeffs<T>.back()));
                }
            }(indices, x);
        }
    }

    template <floating_point T, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL horner(basic_simd<T, A> x) noexcept {
        using simd = basic_simd<T, A>;
        return []<size_t I = sizeof...(Vs) - 2>(
            this auto self, simd result, simd x, immediate<I> = {}) {
            if constexpr (I > 0) {
                return self(
                    dx::fmadd(result, x, coeffs<T>[imm<I>]), x, imm<I - 1>);
            } else {
                return dx::fmadd(result, x, coeffs<T>[imm<I>]);
            }
        }
        (dx::broadcast<A>(coeffs<T>.back()), x);
    }

public:
    template <floating_point T, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_simd<T, A> x) noexcept {
        if constexpr (sizeof...(Vs) <= 5) {
            return horner(x);
        } else {
            return estrin(x);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t) noexcept {
        constexpr auto val = front(Vs...);
        return val;
    }
};
} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END
