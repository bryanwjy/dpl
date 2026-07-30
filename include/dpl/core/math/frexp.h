// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/floating_point_simd.h"
#include "dpl/core/math/details/frexp_options.h"
#include "dpl/core/math/details/ilogb.h"
#include "dpl/core/math/details/ldexp.h"
#include "dpl/core/math/ldexp.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/exponent_bias.h"
#  include "dpl/core/immediate/constants/mantissa_width.h"
#  include "dpl/core/immediate/constants/min_value.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/simd_native_tuple.h"
#  include "dpl/std/concepts/integral.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::frexp_options {
using sign_src_t = fmath::frexp_sign_t<fmath::frexp_sign::src>;
using sign_zero_t = fmath::frexp_sign_t<fmath::frexp_sign::zero>;
using sign_nan_t = fmath::frexp_sign_t<fmath::frexp_sign::nan>;
using standard_t = fmath::frexp_interval_t<fmath::frexp_interval::standard>;
using binade_t = fmath::frexp_interval_t<fmath::frexp_interval::binade>;
using extended_t = fmath::frexp_interval_t<fmath::frexp_interval::extended>;
using reduced_t = fmath::frexp_interval_t<fmath::frexp_interval::reduced>;
using default_t = fmath::frexp_options_t<fmath::frexp_sign::src,
    fmath::frexp_interval::standard>;
inline constexpr sign_src_t sign_src{};
inline constexpr sign_zero_t sign_zero{};
inline constexpr sign_nan_t sign_nan{};
inline constexpr standard_t standard{};
inline constexpr binade_t binade{};
inline constexpr extended_t extended{};
inline constexpr reduced_t reduced{};
inline constexpr default_t default_v{};
} // namespace datapar::frexp_options

namespace datapar::internal {
template <simd_vector T0, simd_vector T1 = T0>
struct frexp_pair {
    T0 fr;
    T1 exp;

    template <size_t I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr __DPL tuple_element_t<I, frexp_pair> get(
        frexp_pair const& result) noexcept
    requires (I < 2)
    {
        if constexpr (I == 0) {
            return result.fr;
        } else {
            return result.exp;
        }
    }
};

void frexp(...) noexcept = delete;

struct frexp_t : public math_operation_base<frexp_t> {
    using operation_base<frexp_t>::operator();

    template <simd_vector T, frexp_options_type O = frexp_options::default_t>
    requires default_initializable<decay_t<T>> &&
        (!scalable_abi<simd_abi_type_t<T>>) &&
        cpo_invocable<frexp_t, T, decay_t<T>&, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr frexp_pair<decay_t<T>>
        DPL_VECTORCALL operator()(T&& val,
            O opt = frexp_options::default_v) noexcept(canonical_vector<T>) {
        using A = simd_abi_type_t<T>;
        frexp_pair<decay_t<T>> pair;
        pair.fr = operator()(__DPL forward<T>(val), pair.exp, opt);
        return pair;
    }

    // scalable abis
    template <canonical_vector T,
        frexp_options_type O = frexp_options::default_t>
    requires same_as<T, simd_native_type_t<T>> &&
        cpo_invocable<frexp_t, T, T&, O> && requires(T val) {
            typename simd_native_tuple_t<T, 2>;
            make_tuple(internal::abi<T>, val, val);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_native_tuple_t<T, 2>
        DPL_VECTORCALL operator()(
            T val, O opt = frexp_options::default_v) noexcept {
        using A = simd_abi_type_t<T>;
        static_assert(scalable_abi<A>);
        T exp = val;
        auto fr = operator()(val, exp, opt);
        return make_tuple(internal::abi<A>, fr, exp);
    }
};

template <>
struct operation_signature<frexp_t> {
    static consteval void operator()(simd_vector auto&&, simd_vector auto&&,
        frexp_options_type auto) noexcept {}
    static consteval void operator()(
        simd_vector auto&&, simd_vector auto&&) noexcept {}
};

template <typename T, typename Exp, typename O, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_frexp = requires(T val, Exp& exp, O opt) {
    { frexp(internal::abi<A>, val, exp, opt) } -> equivalent_vector_with<T>;
};

template <>
struct canonical_impl<frexp_t> {
    template <typename T>
    using viexp_t DPL_NODEBUG =
        rebind_simd_t<T, signed_representation_t<simd_element_type_t<T>>>;

    template <canonical_vector T,
        frexp_options_type O = frexp_options::default_t>
    requires unqualified_canonical_frexp<T, T&, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, type_identity_t<T>& exp,
        O opt = frexp_options::default_v) noexcept {
        return frexp(internal::abi<T>, val, exp, opt);
    }

    template <canonical_vector T,
        frexp_options_type O = frexp_options::default_t>
    requires unqualified_canonical_frexp<T, viexp_t<T>&, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        T val, viexp_t<T>& exp, O opt = frexp_options::default_v) noexcept {
        return frexp(internal::abi<T>, val, exp, opt);
    }
};

template <typename T, typename Exp, typename O, typename A = simd_abi_type_t<T>>
concept unqualified_extended_frexp = requires(O opt) {
    {
        frexp(internal::declarg<T>(), internal::declarg<Exp>(), opt)
    } -> equivalent_vector_with<T>;
};

template <>
struct extended_impl<frexp_t> {
    template <simd_vector T, equivalent_vector_with<T> Exp,
        frexp_options_type O = frexp_options::default_t>
    requires (extended_vector<T> || extended_vector<Exp>) &&
        unqualified_extended_frexp<T, Exp, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        T&& val, Exp&& exp, O opt = frexp_options::default_v) noexcept {
        return frexp(__DPL forward<T>(val), exp, opt);
    }

    template <simd_vector T, simd_vector Exp,
        frexp_options_type O = frexp_options::default_t>
    requires same_as<signed_representation_t<simd_element_type_t<T>>,
                 simd_element_type_t<Exp>> &&
        same_abi_as<simd_abi_type_t<T>, simd_abi_type_t<Exp>> &&
        (extended_vector<T> || extended_vector<Exp>) &&
        unqualified_extended_frexp<T, Exp, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        T&& val, Exp&& exp, O opt = frexp_options::default_v) noexcept {
        return frexp(__DPL forward<T>(val), exp, opt);
    }
};

template <>
struct fallback_impl<frexp_t> {
private:
    template <floating_point_like E>
    static constexpr auto normalizer = []() {
        if constexpr (dx::digits_v<E> >= dx::digits_v<float>) {
            return static_cast<E>(0x1.p64);
        } else {
            return static_cast<E>(0x1.p12);
        }
    }();

    template <typename E>
    using iexp_t = signed_representation_t<E>;

    template <floating_point_like E>
    static constexpr iexp_t<E> subnormal_offset = []() {
        if constexpr (dx::digits_v<E> >= dx::digits_v<float>) {
            return 64;
        } else {
            return 12;
        }
    }();

    template <canonical_vector T, frexp_options_type Opt,
        typename E = simd_element_type_t<T>>
    requires floating_point_like<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL impl(
        T val, rebind_simd_t<T, iexp_t<E>>& iexp, Opt) noexcept
    requires (Opt::contains(frexp_options::reduced))
    {
        using A = simd_abi_type_t<T>;
        constexpr auto fourthirds = static_cast<E>(1.0 / 0.75);
        auto const vfourthirds = dx::broadcast<E, A>(fourthirds);
        auto const issubnormal = [](auto val) {
            if constexpr (Opt::contains(frexp_options::sign_nan)) {
                return dx::cmplt(val, dx::min_value);
            } else {
                return dx::cmpeq(dx::bwand(val, dx::infinity), dx::zero);
            }
        }(val);

        auto const vnormalizer = dx::broadcast<A>(normalizer<E>);
        val = dx::multiply(val, issubnormal, val, vnormalizer);
        if constexpr (Opt::contains(frexp_options::sign_zero)) {
            val = dx::abs(val);
        }

        iexp = fmath::ilogb(
            fmath::compliance::unsafe, dx::multiply(val, vfourthirds));
        auto const offset = dx::broadcast<A>(subnormal_offset<E>);
        iexp = dx::subtract(iexp, issubnormal, iexp, offset);

        auto fr = fmath::ldexp(fmath::compliance::unsafe, val,
            dx::isfinite(val), val, dx::negate(iexp));
        if constexpr (Opt::contains(frexp_options::sign_nan)) {
            return dx::select(dx::cmpgt(val, dx::zero), fr, dx::all_bits);
        } else {
            return fr;
        }
    }

    template <canonical_vector T, frexp_options_type Opt,
        typename E = simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL impl(
        T val, rebind_simd_t<T, iexp_t<E>>& iexp, Opt) noexcept
    requires (Opt::contains(frexp_options::standard))
    {
        auto const issubnormal = [](auto val) {
            if constexpr (Opt::contains(frexp_options::sign_nan)) {
                return dx::cmplt(val, dx::min_value);
            } else {
                return dx::cmpeq(dx::bwand(val, dx::exponent_bits), dx::zero);
            }
        }(val);

        constexpr auto exp_bits = __DPL bit_cast<iexp_t<E>>(
            floating_point_traits<E>::exponent_mask);
        constexpr iexp_t<E> magic_v =
            floating_point_traits<E>::exponent_bias - 1;
        constexpr auto exp_shift =
            __DPL countr_zero(floating_point_traits<E>::exponent_mask);
        constexpr auto all_exp = __DPL bit_cast<iexp_t<E>>(
            floating_point_traits<E>::exponent_mask >> exp_shift);

        using A = simd_abi_type_t<T>;
        auto const magic = dx::broadcast<A>(magic_v);
        auto const offset = dx::broadcast<A>(subnormal_offset<E>);
        iexp = fmath::ilogb(fmath::compliance::unsafe, val);
        iexp = dx::subtract(iexp, dx::cmpneq(iexp, all_exp), iexp, magic);
        // perform the select for better ILP
        iexp = dx::subtract(iexp, dx::select(issubnormal, offset, dx::zero));

        using bitset_t = bit_representation_t<E>;
        constexpr auto magic_exp = [&]() {
            auto const signctrl = Opt::contains(frexp_options::sign_zero)
                ? ~floating_point_traits<E>::signbit
                : ~bitset_t();

            return __DPL bit_cast<E>(
                (bitset_t(magic_v) << exp_shift) & signctrl);
        }();

        auto const vnormalizer = dx::broadcast<A>(normalizer<E>);
        val = dx::multiply(val, issubnormal, val, vnormalizer);
        auto fr = dx::bwand(val, dx::isfinite(val), val, magic_exp);
        if constexpr (Opt::contains(frexp_options::sign_nan)) {
            return dx::select(dx::cmpgt(val, dx::zero), fr, dx::all_bits);
        } else {
            return fr;
        }
    }

public:
    template <canonical_vector T,
        frexp_options_type Opt = frexp_options::default_t,
        typename E = simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T val,
        rebind_simd_t<T, iexp_t<E>>& iexp,
        Opt opt = frexp_options::default_v) noexcept
    requires floating_point_like<E> &&
        requires { fallback_impl::impl(val, iexp, opt); }
    {
        return fallback_impl::impl(val, iexp, opt);
    }

    template <canonical_vector T,
        frexp_options_type Opt = frexp_options::default_t,
        typename E = simd_element_type_t<T>>
    requires floating_point_like<E> &&
        cpo_invocable<frexp_t, T, rebind_simd_t<T, iexp_t<E>>, Opt>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val,
        type_identity_t<T>& fexp, Opt opt = frexp_options::default_v) noexcept {
        rebind_simd_t<T, iexp_t<E>> iexp;
        auto fr = frexp_t::operator()(val, iexp, opt);
        fexp = dx::element_cast<E>(iexp);
        return fr;
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::frexp_t frexp{};
}
} // namespace datapar

template <datapar::simd_vector T0, datapar::simd_vector T1>
struct tuple_size<datapar::internal::frexp_pair<T0, T1>> :
    size_constant<2zu> {};

template <datapar::simd_vector T0, datapar::simd_vector T1>
struct tuple_element<0zu, datapar::internal::frexp_pair<T0, T1>> :
    type_identity<T0> {};

template <datapar::simd_vector T0, datapar::simd_vector T1>
struct tuple_element<1zu, datapar::internal::frexp_pair<T0, T1>> :
    type_identity<T1> {};

namespace frexp_options = __DPL datapar::frexp_options;
__DPL_DEFAULT_NAMESPACE_END
