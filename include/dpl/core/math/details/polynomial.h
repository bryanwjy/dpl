// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/immediate/constants/digits.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/numbers/floating_point_like.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/type_traits/remove_const.h"
#endif

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace estrin {
struct unintialized_t {};

/**
 * Use a union for easier debugging at constexpr
 */
template <floating_point_like E, simd_abi A>
union optional {
    unintialized_t none;
    basic_vector<E, A> val;
};

/**
 * Union based storage used to reduce register pressure
 * on estrin evalutaions.
 */
template <size_t S, floating_point_like E, simd_abi A>
class vpowers : protected vpowers<S - 1, E, A> {
    using base_type DPL_NODEBUG = vpowers<S - 1, E, A>;

public:
    __DPL_HIDE_FROM_ABI constexpr vpowers() noexcept = default;
    __DPL_HIDE_FROM_ABI explicit constexpr vpowers(
        basic_vector<E, A> x0) noexcept
    requires (S == 0)
        : data{
              .val = x0,
          } {}

    __DPL_HIDE_FROM_ABI explicit constexpr vpowers(
        basic_vector<E, A> x0) noexcept
        : base_type(nullptr, x0)
        , data{.none = {}} {}

    template <integral auto I>
    requires (S > 0 && I <= S)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr basic_vector<E, A> operator[](immediate<I>) const noexcept {
        if constexpr (I == S) {
            return data.val;
        } else {
            return vpowers<I, E, A>::data.val;
        }
    }

    template <integral auto I>
    requires (I <= S && S > 0)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    constexpr void initialize() noexcept {
        vpowers<I, E, A>::data = optional<E, A>{
            .val = dx::multiply(
                vpowers<I - 1, E, A>::data.val, vpowers<I - 1, E, A>::data.val),
        };
    }

protected:
    __DPL_HIDE_FROM_ABI explicit constexpr vpowers(
        decltype(nullptr), basic_vector<E, A> x0) noexcept
    requires (S == 1)
        : base_type(x0)
        , data{.val = dx::multiply(x0, x0)} {}

    __DPL_HIDE_FROM_ABI explicit constexpr vpowers(
        decltype(nullptr) tag, basic_vector<E, A> x0) noexcept
        : base_type(tag, x0)
        , data{.none = {}} {}

    optional<E, A> data;
};

template <floating_point_like E, simd_abi A>
class vpowers<static_cast<size_t>(-1), E, A> {};

} // namespace estrin

template <floating_point_like auto V0, floating_point_like auto... Vs>
class polynomial {
private:
    static_assert(sizeof...(Vs) >= 1);
    template <floating_point_like T>
    struct coeffs_t {
        template <integral auto I>
        requires (I == 0)
        static consteval T operator[](immediate<I>) noexcept {
            return V0;
        }

        template <integral auto I>
        static consteval T operator[](immediate<I>) noexcept {
#if (DPL_HAS_CXX26_EXTENSIONS || DPL_CXX26) && __cpp_pack_indexing >= 202311L
            static_assert(digits_v<T> <=
                digits_v<remove_const_t<decltype(Vs...[I - 1])>>);
            return static_cast<T>(Vs...[I - 1]);
#else
            static_assert((... &&
                (digits_v<T> <= digits_v<remove_const_t<decltype(Vs)>>)));
            using array = T[sizeof...(Vs)];
            return array{Vs...}[I - 1];
#endif
        }

        consteval T back(this coeffs_t self) noexcept {
            return self[imm<sizeof...(Vs)>];
        }

        consteval T front(this coeffs_t self) noexcept { return V0; }
    };

    template <floating_point_like T>
    static constexpr coeffs_t<T> coeffs{};

    static constexpr auto degree = sizeof...(Vs);
    static constexpr size_t depth = __DPL bit_width(degree) - 1;

    template <floating_point_like E, simd_abi A, typename Powers,
        size_t B = 0zu, size_t L = depth>
    requires same_as<decay_t<Powers>, fmath::estrin::vpowers<depth, E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto eval_estrin(
        Powers&& x, immediate<B> = {}, immediate<L> = {}) noexcept {
        constexpr auto S = 1 << L; // stride for the next level
        if constexpr (L == 0) {
            // Leaf
            if constexpr (B + S <= degree) {
                return dx::muladd(
                    x[imm<L>], coeffs<E>[imm<B + S>], coeffs<E>[imm<B>]);
            } else {
                return dx::broadcast<A>(coeffs<E>[imm<B>]);
            }
        } else if constexpr (B + S <= degree) {
            auto const right = eval_estrin<E, A>(x, imm<B>, imm<L - 1>);
            auto const left = eval_estrin<E, A>(x, imm<B + S>, imm<L - 1>);
            if constexpr (B == 2 * S) {
                // Late initialization of squares for powers greater than 2
                // Pray that OOO execution can hide it's latency
                x.template initialize<L + 1>();
            }

            // x[L] => x^2^L
            return dx::muladd(x[imm<L>], left, right);
        } else {
            if constexpr (B == 2 * S) {
                // Late initialization of squares for powers greater than 2
                // Pray that OOO execution can hide it's latency
                x.template initialize<L + 1>();
            }

            return eval_estrin<E, A>(x, imm<B>, imm<L - 1>);
        }
    }

    template <floating_point_like E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL eval_estrin(
        basic_vector<E, A> x) noexcept {
        // Compiler Explorer: https://godbolt.org/z/nsvsYnez1
        // Evaluates estrin with lower register pressure by deferring
        // the square operation as late as possible, using the FMA dependency
        // chain to hide it's latency.
        // This results in lower register pressures and register
        // lifetimes reductions for large polynomials, e.g. 18th degree, on
        // clang
        return eval_estrin<E, A>(estrin::vpowers<depth, E, A>(x));
    }

    template <floating_point_like E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL eval_horner(
        basic_vector<E, A> x) noexcept {
        using simd = basic_vector<E, A>;
        return []<int I = sizeof...(Vs) - 1>(
            this auto self, simd result, simd x, immediate<I> = {}) {
            if constexpr (I > 0) {
                return self(
                    dx::muladd(result, x, coeffs<E>[imm<I>]), x, imm<I - 1>);
            } else {
                return dx::muladd(result, x, coeffs<E>[imm<I>]);
            }
        }
        (dx::broadcast<A>(coeffs<E>.back()), x);
    }

public:
    template <floating_point_like T, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<T, A> x) noexcept {
        if constexpr (degree < 6) {
            return eval_horner(x);
        } else {
            return eval_estrin(x);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t) noexcept {
        return V0;
    }
};
} // namespace datapar::fmath
__DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
