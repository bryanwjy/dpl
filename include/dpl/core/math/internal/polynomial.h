// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fma.h"
#if !DPL_MODULES
#  include "dpl/core/basic/basic_simd.h"
#  include "dpl/core/basic/immediate.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/utility/forward.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <floating_point T, T... Vs>
struct polynomial {
    static_assert(sizeof...(Vs) >= 2);
    static constexpr struct coeffs_t {
        template <size_t I>
        static consteval T operator[](immediate<I>) noexcept {
#if __cpp_pack_indexing >= 202311L & (DPL_CXX26 | DPL_COMPILER_CLANG)
#  if DPL_COMPILER_CLANG & !DPL_CXX26
            DPL_DISABLE_WARNING_PUSH()
            DPL_DISABLE_WARNING("-Wc++26-extensions")
#  endif
            return Vs...[I];
#  if DPL_COMPILER_CLANG & !DPL_CXX26
            DPL_DISABLE_WARNING_POP()
#  endif
#else
            using array = float[sizeof...(Vs)];
            return array{Vs...}[I];
#endif
        }

        consteval T back(this coeffs_t self) noexcept {
            return self[imm<sizeof...(Vs) - 1>];
        }
    } coeffs{};

private:
    template <simd_abi A, same_as<basic_simd<T, A>> T0,
        same_as<basic_simd<T, A>>... Ts>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        estrin(basic_simd<T, A> x, T0 t0, Ts... ts) noexcept {
        using simd = basic_simd<T, A>;
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

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL estrin(basic_simd<T, A> x) noexcept {
        // Can define more if needed, but for now, limit to 8
        static_assert(sizeof...(Vs) <= 8);
        if constexpr (sizeof...(Vs) == 8) {
            static constexpr size_t size = sizeof...(Vs) / 2;
            static constexpr make_index_sequence<size> indices{};
            return []<size_t... Is>(index_sequence<Is...>, auto x) {
                constexpr polynomial<T, coeffs[imm<Is>]...> first{};
                constexpr polynomial<T, coeffs[imm<size + Is>]...> second{};
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
                        dx::fmadd(coeffs[imm<Is * 2 + 1>], x,
                            coeffs[imm<Is * 2>])...);
                } else {
                    return estrin(x,
                        dx::fmadd(
                            coeffs[imm<Is * 2 + 1>], x, coeffs[imm<Is * 2>])...,
                        dx::broadcast<A>(coeffs.back()));
                }
            }(indices, x);
        }
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL horner(basic_simd<T, A> x) noexcept {
        using simd = basic_simd<T, A>;
        return []<size_t I = sizeof...(Vs) - 2>(
            this auto self, simd result, simd x, immediate<I> = {}) {
            if constexpr (I > 0) {
                return self(
                    dx::fmadd(result, x, coeffs[imm<I>]), x, imm<I - 1>);
            } else {
                return dx::fmadd(result, x, coeffs[imm<I>]);
            }
        }
        (dx::broadcast<simd>(coeffs.back()), x);
    }

public:
    template <simd_abi A>
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
        return coeffs[imm<0>];
    }
};
} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END
