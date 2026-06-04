// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/ilogb.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/ldexp.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/integral_simd.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/exponent_bias.h"
#  include "dpl/core/constants/mantissa_width.h"
#  include "dpl/core/constants/min_value.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

namespace internal {

enum class fr_sign {
    copy,
    abs,
    positive,
};

enum class fr_interval {
    canonical, // [1,2)
    cmath,     // [0.5,1)
    wide,      // [0.5,2)
    reduced    // [0.75,1.5)
};

enum class frexp_type {
    integral,
    floating,
};

template <different_from<fr_sign> R>
consteval bool operator==(fr_sign, R) noexcept {
    return false;
}

template <different_from<fr_interval> R>
consteval bool operator==(fr_interval, R) noexcept {
    return false;
}

template <different_from<frexp_type> R>
consteval bool operator==(frexp_type, R) noexcept {
    return false;
}

template <scoped_enumeration auto... Vs>
requires (sizeof...(Vs) > 0 && sizeof...(Vs) <= 3)
struct frexp_options_t {
    explicit consteval frexp_options_t() noexcept = default;

    consteval bool operator==(frexp_options_t) noexcept { return true; }
    consteval bool operator!=(frexp_options_t) noexcept { return false; }

    template <fr_interval I>
    requires (... && different_from<fr_interval, decltype(Vs)>) &&
        (sizeof...(Vs) < 3)
    consteval frexp_options_t<Vs..., I> operator|(
        this frexp_options_t, frexp_options_t<I>) noexcept {
        return frexp_options_t<Vs..., I>{};
    }

    template <fr_sign I>
    requires (... && different_from<fr_sign, decltype(Vs)>) &&
        (sizeof...(Vs) < 3)
    consteval frexp_options_t<Vs..., I> operator|(
        this frexp_options_t, frexp_options_t<I>) noexcept {
        return frexp_options_t<Vs..., I>{};
    }

    template <frexp_type I>
    requires (... && different_from<frexp_type, decltype(Vs)>) &&
        (sizeof...(Vs) < 3)
    consteval frexp_options_t<Vs..., I> operator|(
        this frexp_options_t, frexp_options_t<I>) noexcept {
        return frexp_options_t<Vs..., I>{};
    }

    template <fr_interval I>
    requires (... && different_from<fr_interval, decltype(Vs)>) &&
        (sizeof...(Vs) == 2)
    friend consteval frexp_options_t<I, Vs...> operator|(
        frexp_options_t<I>, frexp_options_t) noexcept {
        return frexp_options_t<I, Vs...>{};
    }

    template <fr_sign I>
    requires (... && different_from<fr_sign, decltype(Vs)>) &&
        (sizeof...(Vs) == 2)
    friend consteval frexp_options_t<I, Vs...> operator|(
        frexp_options_t<I>, frexp_options_t) noexcept {
        return frexp_options_t<I, Vs...>{};
    }

    template <frexp_type I>
    requires (... && different_from<frexp_type, decltype(Vs)>) &&
        (sizeof...(Vs) == 2)
    friend consteval frexp_options_t<I, Vs...> operator|(
        frexp_options_t<I>, frexp_options_t) noexcept {
        return frexp_options_t<I, Vs...>{};
    }

    template <fr_sign V>
    static consteval bool has(
        frexp_options_t<V> = frexp_options_t<V>{}) noexcept {
        if constexpr ((... || (same_as<decltype(Vs), fr_sign>))) {
            return (... || (Vs == V));
        } else {
            return V == fr_sign::copy;
        }
    }

    template <fr_interval V>
    static consteval bool has(
        frexp_options_t<V> = frexp_options_t<V>{}) noexcept {
        if constexpr ((... || (same_as<decltype(Vs), fr_interval>))) {
            return (... || (Vs == V));
        } else {
            return V == fr_interval::cmath;
        }
    }

    template <frexp_type V>
    static consteval bool has(
        frexp_options_t<V> = frexp_options_t<V>{}) noexcept {
        if constexpr ((... || (same_as<decltype(Vs), frexp_type>))) {
            return (... || (Vs == V));
        } else {
            return V == frexp_type::integral;
        }
    }

    static constexpr auto type() noexcept {
        if (frexp_options_t::template has<frexp_type::integral>()) {
            return frexp_options_t<frexp_type::integral>{};
        } else {
            return frexp_options_t<frexp_type::floating>{};
        }
    };
};

using frexp_copysign_t DPL_NODEBUG = frexp_options_t<fr_sign::copy>;
using frexp_abs_t DPL_NODEBUG = frexp_options_t<fr_sign::abs>;
using frexp_positive_t DPL_NODEBUG = frexp_options_t<fr_sign::positive>;
using frexp_canonical_t DPL_NODEBUG = frexp_options_t<fr_interval::canonical>;
using frexp_cmath_t DPL_NODEBUG = frexp_options_t<fr_interval::cmath>;
using frexp_wide_t DPL_NODEBUG = frexp_options_t<fr_interval::wide>;
using frexp_reduced_t DPL_NODEBUG = frexp_options_t<fr_interval::reduced>;
using frexp_integral_t DPL_NODEBUG = frexp_options_t<frexp_type::integral>;
using frexp_floating_point_t DPL_NODEBUG =
    frexp_options_t<frexp_type::floating>;

template <typename>
inline constexpr bool is_frexp_options = false;

template <enumeration auto... Vs>
inline constexpr bool is_frexp_options<frexp_options_t<Vs...>> = true;
} // namespace internal

DPL_EXPORT inline constexpr internal::frexp_copysign_t frexp_copysign{};
DPL_EXPORT inline constexpr internal::frexp_abs_t frexp_abs{};
DPL_EXPORT inline constexpr internal::frexp_positive_t frexp_positive{};
DPL_EXPORT inline constexpr internal::frexp_canonical_t frexp_canonical{};
DPL_EXPORT inline constexpr internal::frexp_cmath_t frexp_cmath{};
DPL_EXPORT inline constexpr internal::frexp_wide_t frexp_wide{};
DPL_EXPORT inline constexpr internal::frexp_reduced_t frexp_reduced{};
DPL_EXPORT inline constexpr internal::frexp_integral_t frexp_integral{};
DPL_EXPORT inline constexpr internal::frexp_floating_point_t frexp_floating_point{};
DPL_EXPORT inline constexpr auto frexp_default =
    frexp_copysign | frexp_cmath | frexp_integral;

template <typename T>
concept is_frexp_type = same_as<T, internal::frexp_integral_t> ||
    same_as<T, internal::frexp_floating_point_t>;

DPL_EXPORT template <internal::floating_point_simd T,
    is_frexp_type auto E = frexp_integral>
struct frexp_result {
    T fr;
    T exp;
};

DPL_EXPORT template <internal::floating_point_simd T>
requires requires {
    typename signed_representation_t<typename T::value_type>;
    typename rebind_simd_t<T, signed_representation_t<typename T::value_type>>;
} &&
    requires(T fr,
        rebind_simd_t<T, signed_representation_t<typename T::value_type>> exp) {
        datapar::ldexp(fr, exp);
    }
struct frexp_result<T, frexp_integral> {
    T fr;
    rebind_simd_t<T, signed_representation_t<typename T::value_type>> exp;
};

DPL_EXPORT template <typename T>
concept frexp_options = internal::is_frexp_options<T>;
} // namespace datapar

namespace datapar::internal {
template <typename L, typename R>
struct make_frexp {
    using type DPL_NODEBUG = frexp_result<L, frexp_integral>;
};
template <typename L, typename R>
using make_frexp_t DPL_NODEBUG = typename make_frexp<L, R>::type;

template <typename L, typename R>
requires (R::has(frexp_floating_point))
struct make_frexp<L, R> {
    using type DPL_NODEBUG = frexp_result<L, frexp_floating_point>;
};

template <typename T, typename L, typename R>
concept frexp_result_type = requires {
    typename make_frexp_t<L, R>;
    requires equivalent_simd_as<decltype(T::fr), decltype(L::fr)>;
    requires equivalent_simd_as<decltype(T::exp),
        decltype(make_frexp_t<L, R>::exp)>;
};

void frexp(...) noexcept = delete;

struct frexp_t;

template <typename T, typename O, typename A = typename T::abi_type>
concept unqualified_canonical_frexp = requires(T val, O opt) {
    { frexp(internal::abi<A>, val, opt) } -> frexp_result_type<T, O>;
};

template <typename T, typename O, typename A = typename T::abi_type>
concept unqualified_extended_frexp = requires(T val, O opt) {
    { frexp(val, opt) } -> frexp_result_type<T, O>;
};

template <typename T, typename O>
concept expression_frexp = simd_expression<T> &&
    regular_invocable<frexp_t, simd_expression_result_t<T>, O>;

template <typename T, typename O>
concept decayable_frexp =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<frexp_t, canonical_type_t<T>, O>;

template <typename T, typename O, typename A = typename T::abi_type>
concept extended_frexp = unqualified_extended_frexp<T, O, A> ||
    expression_frexp<T, O> || decayable_frexp<T, O>;

struct frexp_t {
private:
    template <floating_point E>
    static constexpr auto denormalizer = []() {
        if constexpr (dx::digits_v<E> >= dx::digits_v<float>) {
            return static_cast<E>(0x1.p64);
        } else {
            return static_cast<E>(0x1.p12);
        }
    }();
    template <floating_point E>
    static constexpr auto subnormal_offset = []() {
        if constexpr (dx::digits_v<E> >= dx::digits_v<float>) {
            return 64;
        } else {
            return 12;
        }
    }();

    template <floating_point E, simd_abi A, frexp_options Opt>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val, Opt) noexcept
    requires (Opt::has(frexp_reduced))
    {
        using result_type = make_frexp_t<basic_vector<E, A>, Opt>;
        constexpr auto fourthirds = dx::broadcast<E, A>(1.0 / 0.75);
        auto const issubnormal = [](auto val) {
            if constexpr (Opt::has(frexp_positive)) {
                return val < dx::min_value;
            } else {
                return (val & dx::exponent_bits) == dx::zero;
            }
        }(val);
        auto const dval = [](auto issubnormal, auto val) {
            auto const dval =
                dx::select(issubnormal, val * denormalizer<E>, val);
            if constexpr (Opt::has(frexp_abs)) {
                return dx::abs(val);
            } else {
                return val;
            }
        }(issubnormal, val);

        auto const exp =
            [](auto issubnormal, auto exp) {
                return dx::select(issubnormal, exp - subnormal_offset<E>, exp);
            }(issubnormal,
                fmath::ilogb(fmath::compliance::unsafe, dval * fourthirds));
        auto const fr = [](auto val, auto fr) {
            if constexpr (Opt::has(frexp_positive)) {
                return dx::select(val < dx::zero, dx::all_bits, fr);
            } else {
                return fr;
            }
        }(fmath::ldexp(fmath::compliance::unsafe, dval, -exp));
        if constexpr (Opt::has(frexp_integral)) {
            return result_type{
                .fr = fr,
                .exp = exp,
            };
        } else {
            return result_type{
                .fr = fr,
                .exp = dx::element_cast<E>(exp),
            };
        }
    }

    template <floating_point E, simd_abi A, frexp_options Opt>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val, Opt) noexcept
    requires (Opt::has(frexp_cmath))
    {
        using result_type = make_frexp_t<basic_vector<E, A>, Opt>;
        auto const issubnormal = [](auto val) {
            if constexpr (Opt::has(frexp_positive)) {
                return val < dx::min_value;
            } else {
                return (val & dx::exponent_bits) == dx::zero;
            }
        }(val);
        using int_type = dx::signed_representation_t<E>;
        constexpr auto exp_bits =
            __DPL bit_cast<int_type>(dx::exponent_bits_v<E>);
        constexpr auto magic =
            dx::broadcast<int_type, A>(exponent_bias_v<E> - 1);
        constexpr auto magic_exp =
            __DPL bit_cast<E>(magic << dx::mantissa_width_v<E>);

        auto const exp = [&]() {
            auto const mexp =
                dx::reinterpret<int_type>(val & dx::exponent_bits);
            auto const exp_offset = dx::select(
                mexp == dx::zero || mexp == exp_bits, dx::zero, magic);
            auto const exp = (mexp >> imm<dx::mantissa_width_v<E>>)-exp_offset;

            return dx::select(issubnormal, exp - subnormal_offset<E>, exp);
        }();
        auto const fr = [&]() {
            auto const fr = (val & ~exponent_bits) | magic_exp;
            if constexpr (Opt::has(frexp_positive)) {
                return dx::select(val < dx::zero, dx::all_bits, fr);
            } else {
                return fr;
            }
        }();
        if constexpr (Opt::has(frexp_integral)) {
            return result_type{
                .fr = fr,
                .exp = exp,
            };
        } else {
            return result_type{
                .fr = fr,
                .exp = dx::element_cast<E>(exp),
            };
        }
    }

public:
    template <simd_vector T>
    requires floating_point<typename T::value_type>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        return operator()(val, frexp_copysign | frexp_cmath | frexp_integral);
    }

    template <simd_abi A, simd_element_for<A> E, frexp_options Opt>
    requires floating_point<E> &&
        (Opt() == frexp_cmath || Opt() == frexp_reduced)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, Opt opt) noexcept {
        if constexpr (unqualified_canonical_frexp<basic_vector<E, A>, Opt>) {
            if consteval {
                return fallback(val);
            } else {
                return frexp(internal::abi<A>, val, opt);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E, frexp_options Opt>
    requires (!floating_point<E>) &&
        unqualified_canonical_frexp<basic_vector<E, A>, Opt>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, Opt opt) noexcept {
        return frexp(internal::abi<A>, val, opt);
    }

    template <extended_vector T, frexp_options Opt>
    requires extended_frexp<T, Opt>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, Opt opt) noexcept {
        if constexpr (unqualified_extended_frexp<T, Opt>) {
            return frexp(val, opt);
        } else if constexpr (expression_frexp<T, Opt>) {
            return operator()(dx::evaluate(val), opt);
        } else {
            return operator()(dx::to_canonical(val), opt);
        }
    }
};

} // namespace datapar::internal

namespace datapar {

DPL_EXPORT template <internal::floating_point_simd Fr,
    common_size_simd_with<Fr> Exp, frexp_options Opt = decltype(frexp_default)>
requires (Opt::has(frexp_integral) && internal::integral_simd<Exp>) ||
    (Opt::has(frexp_floating_point) && same_as<Exp, Fr>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto make_frexp_result(Fr fr, Exp exp, Opt = frexp_default) noexcept {
    return frexp_result<Fr, Opt::type()>{
        .fr = fr,
        .exp = exp,
    };
}

inline namespace cpo {
DPL_EXPORT inline constexpr internal::frexp_t frexp{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
