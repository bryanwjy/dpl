// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.combo_multiply;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_math_operation auto mulop, dpp::simd_abi A>
class combo_multiply {
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr E expected_op(E lhs, E mid, E rhs) noexcept {
        if constexpr (!same_as<double, E>) {
            return expected_op<double>(lhs, mid, rhs);
        } else if constexpr (mulop == dpp::muladd) {
            return lhs * mid + rhs;
        } else if constexpr (mulop == dpp::mulsub) {
            return lhs * mid - rhs;
        } else if constexpr (mulop == dpp::nmuladd) {
            return rhs - lhs * mid;
        } else if constexpr (mulop == dpp::nmulsub) {
            return -rhs - lhs * mid;
        } else if constexpr (mulop == dpp::mulacc) {
            return lhs + mid * rhs;
        } else if constexpr (mulop == dpp::mulsac) {
            return mid * rhs - lhs;
        } else if constexpr (mulop == dpp::nmulacc) {
            return lhs - mid * rhs;
        } else {
            static_assert(mulop == dpp::nmulsac);
            return -lhs - mid * rhs;
        }
    }

    template <typename E>
    static constexpr auto max = []() {
        auto bias = dpl::floating_point_traits<E>::exponent_bias;
        auto exp = dpl::floating_point_traits<E>::exponent_mask;
        auto shift = dpl::countr_zero(exp);
        // Use quad root instead of square root
        auto qrt =
            dpl::bitset<dpl::type_bit_v<E>>(
                ((dpl::to_underlying(exp >> shift) - bias) >> 2) + bias - 1)
            << shift;
        return dpl::bit_cast<E>(qrt);
    }();
    template <typename E>
    static constexpr auto min = -max<E>;

    template <typename E>
    static constexpr auto mixmask() noexcept
    requires (mulop == dpp::muladdsub || mulop == dpp::muladdsac)
    {
        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto lanes = abi_traits<E>::size();
            using bitset_t = dpl::bitset<lanes>;
            using alt_cmask_t =
                dpp::const_mask<lanes, test::repeat_byte<lanes>(0x55u)>;
            return alt_cmask_t();
        } else {
            return dpp::cmpeq(
                dpp::bwand(dpp::lane_index<A, E>(), dpp::one), dpp::zero);
        }
    }

    template <typename E>
    static constexpr auto mixmask() noexcept
    requires (mulop == dpp::mulsubadd || mulop == dpp::mulsubacc)
    {
        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto lanes = abi_traits<E>::size();
            using bitset_t = dpl::bitset<lanes>;
            using alt_cmask_t =
                dpp::const_mask<lanes, test::repeat_byte<lanes>(0xAAu)>;
            return alt_cmask_t();
        } else {
            return dpp::cmpeq(
                dpp::bwand(dpp::lane_index<A, E>(), dpp::one), dpp::one);
        }
    }

public:
    template <rng_like Rng, dpl::floating_point_like... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return combo_multiply::template run<E>(engine);
            },
            pack);
    }

    template <dpl::floating_point_like E, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires (mulop == dpp::muladdsub || mulop == dpp::mulsubadd ||
        mulop == dpp::muladdsac || mulop == dpp::mulsubacc)
    {
        dpl::test::array_generator<A, E> const data_generator(min<E>, max<E>);
        dpl::test::scalar_generator<E> const src_generator(
            max<E> * max<E> * max<E>, dpp::max_value_v<E>);
        constexpr auto expected_op = [](vec_t<E> vlhs, vec_t<E> vmid,
                                         vec_t<E> vrhs) noexcept {
            if constexpr (mulop == dpp::muladdsub || mulop == dpp::mulsubadd) {
                return dpp::select(mixmask<E>(), dpp::muladd(vlhs, vmid, vrhs),
                    dpp::mulsub(vlhs, vmid, vrhs));
            } else {
                return dpp::select(mixmask<E>(), dpp::mulacc(vlhs, vmid, vrhs),
                    dpp::mulsac(vlhs, vmid, vrhs));
            }
        };
        for (auto i = 0zu; i < 4; ++i) {
            auto const lhs = data_generator(engine);
            auto const mid = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const src = src_generator(engine);
            auto const vlhs = dpp::load<E, A>(lhs.data());
            auto const vmid = dpp::load<E, A>(mid.data());
            auto const vrhs = dpp::load<E, A>(rhs.data());

            auto const vexpected = expected_op(vlhs, vmid, vrhs);
            auto const vactual = mulop(vlhs, vmid, vrhs);

            assert(dpp::all_of(vexpected == vactual));
            dpl::test::ternary_assignment<A>::template test_masked<E>(
                lhs, mid, rhs, mulop, src);

            if constexpr (dpp::is_simd_canonical_invocable<vec_t<E>, vec_t<E>,
                              vec_t<E>>(dpp::fmadd)) {
                if not consteval {
                    // catostrophic cancellation test
                    {
                        constexpr auto digitsm1 = dpl::countr_zero(
                            floating_point_traits<E>::exponent_mask);
                        // This is equivalent to 2^0 * 2^-digitsm1
                        constexpr auto small_exp =
                            floating_point_traits<E>::exponent_bias - digitsm1;
                        using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
                        constexpr auto small =
                            dpl::bit_cast<E>(bitset_t(small_exp) << digitsm1);
                        auto const vone = dpp::broadcast<A, E>(1);
                        auto const vsmall = dpp::broadcast<A, E>(small);
                        auto const a = dpp::add(vone, vsmall);
                        auto const b = dpp::subtract(vone, vsmall);
                        auto const c = dpp::negate(vone, mixmask<E>(), vone);
                        if constexpr (mulop == dpp::fmaddsub ||
                            mulop == dpp::fmsubadd) {
                            assert(dpp::all_of(
                                expected_op(a, b, c) == mulop(a, b, c)));
                        } else {
                            assert(dpp::all_of(
                                expected_op(c, a, b) == mulop(c, a, b)));
                        }
                    }
                    {
                        auto const vtwo = dpp::broadcast<A, E>(2);
                        auto const vlarge =
                            dpp::broadcast<A, E>(dpp::max_value);
                        auto const c =
                            dpp::negate(vlarge, mixmask<E>(), vlarge);
                        if constexpr (mulop == dpp::fmaddsub ||
                            mulop == dpp::fmsubadd) {
                            assert(dpp::all_of(expected_op(vlarge, vtwo, c) ==
                                mulop(vlarge, vtwo, c)));
                        } else {
                            assert(dpp::all_of(expected_op(c, vlarge, vtwo) ==
                                mulop(c, vlarge, vtwo)));
                        }
                    }
                }
            }
        }

        return true;
    }

    template <dpl::floating_point_like E, rng_like Rng>
    static constexpr bool run(Rng& engine) {

        dpl::test::array_generator<A, E> const data_generator(min<E>, max<E>);
        dpl::test::scalar_generator<E> const src_generator(
            max<E> * max<E> * max<E>, dpp::max_value_v<E>);

        auto const lhs = data_generator(engine);
        auto const mid = data_generator(engine);
        auto const rhs = data_generator(engine);
        auto const src = src_generator(engine);

        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op<E>(lhs[i], mid[i], rhs[i]);
        }

        dpl::test::ternary_assignment<A>::template test<E>(
            lhs, mid, rhs, mulop, expected);
        dpl::test::ternary_assignment<A>::template test_masked<E>(
            lhs, mid, rhs, mulop, src);

        dpl::test::ternary_assignment<A>::template test<E>(
            1, 1, 1, mulop, expected_op<E>(1, 1, 1));
        if constexpr (mulop == dpp::muladd || mulop == dpp::mulsub ||
            mulop == dpp::nmuladd || mulop == dpp::nmulsub) {
            dpl::test::ternary_assignment<A>::template test<E>(
                1, 1, 0, mulop, expected_op<E>(1, 1, 0));
            auto const a = src_generator(engine), b = src_generator(engine);
            dpl::test::ternary_assignment<A>::template test<E>(
                a, b, 0, mulop, expected_op<E>(a, b, 0));
        } else {
            dpl::test::ternary_assignment<A>::template test<E>(
                0, 1, 1, mulop, expected_op<E>(0, 1, 1));
            auto const a = src_generator(engine), b = src_generator(engine);
            dpl::test::ternary_assignment<A>::template test<E>(
                0, a, b, mulop, expected_op<E>(0, a, b));
        }

        auto const cancellation_tests = []() {
            if constexpr (dpp::is_simd_canonical_invocable<vec_t<E>, vec_t<E>,
                              vec_t<E>>(dpp::fmadd)) {
                auto const range =
                    dpl::popcount(dpl::floating_point_traits<E>::exponent_mask);
                auto const drange = dpl::popcount(
                    dpl::floating_point_traits<double>::exponent_mask);
                auto const precision = dpl::floating_point_traits<E>::digits;
                auto const dprecision =
                    dpl::floating_point_traits<double>::digits;
                if consteval {
                    return drange > drange && dprecision > precision;
                } else {
                    return true;
                }
            } else {
                return false;
            }
        }();

        if (cancellation_tests) {
            // catostrophic cancellation test
            constexpr auto digitsm1 =
                dpl::countr_zero(floating_point_traits<E>::exponent_mask);
            // This is equivalent to 2^0 * 2^-digitsm1
            constexpr auto small_exp =
                floating_point_traits<E>::exponent_bias - digitsm1;
            using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
            constexpr auto small =
                dpl::bit_cast<E>(bitset_t(small_exp) << digitsm1);

            if constexpr (mulop == dpp::muladd) {
                dpl::test::ternary_assignment<A>::template test<E>(
                    1 + small, 1 - small, -1, mulop, -small * small);
                // overflow cancels
                dpl::test::ternary_assignment<A>::template test<E>(
                    dpp::max_value_v<E>, 2, -dpp::max_value_v<E>, mulop,
                    dpp::max_value_v<E>);
            } else if constexpr (mulop == dpp::mulsub) {
                dpl::test::ternary_assignment<A>::template test<E>(
                    1 + small, 1 - small, 1, mulop, -small * small);
                // overflow cancels
                dpl::test::ternary_assignment<A>::template test<E>(
                    dpp::max_value_v<E>, 2, dpp::max_value_v<E>, mulop,
                    dpp::max_value_v<E>);
            } else if constexpr (mulop == dpp::nmuladd) {
                dpl::test::ternary_assignment<A>::template test<E>(
                    1 + small, 1 - small, 1, mulop, small * small);
                // overflow cancels
                dpl::test::ternary_assignment<A>::template test<E>(
                    dpp::max_value_v<E>, 2, dpp::max_value_v<E>, mulop,
                    -dpp::max_value_v<E>);
            } else if constexpr (mulop == dpp::nmulsub) {
                dpl::test::ternary_assignment<A>::template test<E>(
                    1 + small, 1 - small, -1, mulop, small * small);
                // overflow cancels
                dpl::test::ternary_assignment<A>::template test<E>(
                    dpp::max_value_v<E>, 2, -dpp::max_value_v<E>, mulop,
                    -dpp::max_value_v<E>);
            }
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using muladd = combo_multiply<dpp::muladd, A>;
export template <dpp::simd_abi A>
using mulsub = combo_multiply<dpp::mulsub, A>;
export template <dpp::simd_abi A>
using nmuladd = combo_multiply<dpp::nmuladd, A>;
export template <dpp::simd_abi A>
using nmulsub = combo_multiply<dpp::nmulsub, A>;
export template <dpp::simd_abi A>
using muladdsub = combo_multiply<dpp::muladdsub, A>;
export template <dpp::simd_abi A>
using mulsubadd = combo_multiply<dpp::mulsubadd, A>;

export template <dpp::simd_abi A>
using mulacc = combo_multiply<dpp::mulacc, A>;
export template <dpp::simd_abi A>
using mulsac = combo_multiply<dpp::mulsac, A>;
export template <dpp::simd_abi A>
using nmulacc = combo_multiply<dpp::nmulacc, A>;
export template <dpp::simd_abi A>
using nmulsac = combo_multiply<dpp::nmulsac, A>;
export template <dpp::simd_abi A>
using muladdsac = combo_multiply<dpp::muladdsac, A>;
export template <dpp::simd_abi A>
using mulsubacc = combo_multiply<dpp::mulsubacc, A>;

} // namespace dpl::test
