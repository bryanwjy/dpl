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
            return -(rhs + lhs * mid);
        } else if constexpr (mulop == dpp::nmulsub) {
            return rhs - lhs * mid;
        } else if constexpr (mulop == dpp::mulacc) {
            return lhs + mid * rhs;
        } else if constexpr (mulop == dpp::mulsac) {
            return mid * rhs - lhs;
        } else if constexpr (mulop == dpp::nmulacc) {
            return -(lhs + mid * rhs);
        } else {
            static_assert(mulop == dpp::nmulsac);
            return lhs - mid * rhs;
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
            constexpr auto tiled = [lanes] {
                auto const rep =
                    dpl::truncate<8>(dpl::to_bit_representation(0x55u));
                if constexpr (lanes <= 8) {
                    return dpl::truncate<lanes>(rep);
                } else if constexpr (lanes % 8 == 0) {
                    return dpl::bit_tile<lanes / 8>(rep);
                } else {
                    return dpl::truncate<lanes>(
                        dpl::bit_tile<(lanes + 7) / 8>(rep));
                }
            }();
            return dpp::deduce_const_mask_v<tiled>;
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
            constexpr auto tiled = [lanes] {
                auto const rep =
                    dpl::truncate<8>(dpl::to_bit_representation(0xAAu));
                if constexpr (lanes <= 8) {
                    return dpl::truncate<lanes>(rep);
                } else if constexpr (lanes % 8 == 0) {
                    return dpl::bit_tile<lanes / 8>(rep);
                } else {
                    return dpl::truncate<lanes>(
                        dpl::bit_tile<(lanes + 7) / 8>(rep));
                }
            }();
            return dpp::deduce_const_mask_v<tiled>;
        } else {
            return dpp::cmpeq(
                dpp::bwand(dpp::lane_index<A, E>(), dpp::one), dpp::one);
        }
    }

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = dpp::simd_abi_traits<A, E>::size();
        auto const limit = []() {
            if consteval {
                return 4zu;
            } else {
                return 128zu;
            }
        }();

        auto max = lanes >= dpl::type_bit_v<size_t>
            ? limit
            : static_cast<size_t>((1zu << lanes) - 1);
        return linear_counter(max < limit ? max : limit);
    }

    template <typename E>
    static constexpr void run_const_mask_test(auto func) noexcept {
        [&]<size_t I = 0, serialized_mt19937 S = {}>(this auto self,
            dpp::immediate<I> = dpp::imm<I>, mt19937_type<S> = {}) {
            if constexpr (I < test_count<E>().size()) {
                constexpr auto lanes = dpp::simd_abi_traits<A, E>::size();
                constexpr bit_generator<lanes> bitgen;
                constexpr auto pair = S.generate_with(bitgen);
                constexpr auto cmask = dpp::deduce_const_mask_v<pair.value>;

                func(cmask);
                self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
            }
        }
        ();
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
        dpl::test::mask_generator<A, E> const mask_generator;
        constexpr auto expected_op = [](auto vlhs, auto vmid,
                                         auto vrhs) noexcept {
            if constexpr (mulop == dpp::muladdsub || mulop == dpp::mulsubadd) {
                return dpp::select(mixmask<E>(), dpp::muladd(vlhs, vmid, vrhs),
                    dpp::mulsub(vlhs, vmid, vrhs));
            } else {
                return dpp::select(mixmask<E>(), dpp::mulacc(vlhs, vmid, vrhs),
                    dpp::mulsac(vlhs, vmid, vrhs));
            }
        };
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);
        for (auto const _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const mid = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const vlhs = dpp::load<E, A>(lhs.data());
            auto const vmid = dpp::load<E, A>(mid.data());
            auto const vrhs = dpp::load<E, A>(rhs.data());
            auto const vexpected = expected_op(vlhs, vmid, vrhs);
            test::operation_fixture<A>::test(
                vexpected, mulop, vlhs, vmid, vrhs);

            operation_fixture<A>::test_masked(mulop, vlhs, vtrue, vmid, vrhs);
            operation_fixture<A>::test_masked(mulop, vlhs, vfalse, vmid, vrhs);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(mulop, vlhs, mask, vmid, vrhs);
            operation_fixture<A>::test_masked(
                mulop, dpp::zero, mask, vlhs, vmid, vrhs);

            using vec_t = dpl::decay_t<decltype(vlhs)>;
            if constexpr (dpp::is_simd_canonical_invocable<vec_t, vec_t, vec_t>(
                              dpp::fmadd)) {
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
                            test::operation_fixture<A>::test(
                                expected_op(a, b, c), mulop, a, b, c);
                        } else {
                            test::operation_fixture<A>::test(
                                expected_op(c, a, b), mulop, c, a, b);
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
                            test::operation_fixture<A>::test(
                                expected_op(vlarge, vtwo, c), mulop, vlarge,
                                vtwo, c);
                        } else {
                            test::operation_fixture<A>::test(
                                expected_op(c, vlarge, vtwo), mulop, c, vlarge,
                                vtwo);
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
        dpl::test::mask_generator<A, E> const mask_generator;
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);
        for (auto const _ : test_count<E>()) {

            auto const lhs = data_generator(engine);
            auto const mid = data_generator(engine);
            auto const rhs = data_generator(engine);

            auto expected = lhs;
            for (auto i = 0zu; i < expected.size(); ++i) {
                expected[i] = expected_op<E>(lhs[i], mid[i], rhs[i]);
            }

            operation_fixture<A>::test(expected, mulop, lhs, mid, rhs);

            operation_fixture<A>::test_masked(mulop, lhs, vtrue, mid, rhs);
            operation_fixture<A>::test_masked(mulop, lhs, vfalse, mid, rhs);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(mulop, lhs, mask, mid, rhs);
            operation_fixture<A>::test_masked(
                mulop, dpp::zero, mask, lhs, mid, rhs);
        }

        {
            // Possible clang bug? complaining even thopugh it is not nodiscard
            dpl::ignore =
                operation_fixture<A>::test(test::splat(expected_op<E>(1, 1, 1)),
                    mulop, test::splat<E>(1), 1, 1);
        }

        if constexpr (mulop == dpp::muladd || mulop == dpp::mulsub ||
            mulop == dpp::nmuladd || mulop == dpp::nmulsub) {
            dpl::ignore =
                operation_fixture<A>::test(test::splat(expected_op<E>(1, 1, 0)),
                    mulop, test::splat<E>(1), 1, 0);
            auto const a = data_generator.scalar(engine),
                       b = data_generator.scalar(engine);
            dpl::ignore =
                operation_fixture<A>::test(test::splat(expected_op<E>(a, b, 0)),
                    mulop, test::splat(a), b, 0);
        } else {
            dpl::ignore =
                operation_fixture<A>::test(test::splat(expected_op<E>(0, 1, 1)),
                    mulop, test::splat<E>(0), 1, 1);
            auto const a = data_generator.scalar(engine),
                       b = data_generator.scalar(engine);
            dpl::ignore =
                operation_fixture<A>::test(test::splat(expected_op<E>(0, a, b)),
                    mulop, 0, test::splat(a), b);
        }

        using vec_t = dpp::make_canonical_vector_t<E, A>;
        auto const cancellation_tests = []() {
            if constexpr (dpp::is_simd_canonical_invocable<vec_t, vec_t, vec_t>(
                              dpp::fmadd)) {
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
                dpl::ignore =
                    operation_fixture<A>::test(test::splat(-small * small),
                        mulop, test::splat(1 + small), 1 - small, -1);
                // overflow cancels
                dpl::ignore = operation_fixture<A>::test(
                    test::splat(dpp::max_value_v<E>), mulop,
                    test::splat(dpp::max_value_v<E>), 2, -dpp::max_value_v<E>);
            } else if constexpr (mulop == dpp::mulsub) {
                dpl::ignore =
                    operation_fixture<A>::test(test::splat(-small * small),
                        mulop, test::splat(1 + small), 1 - small, 1);
                // overflow cancels
                dpl::ignore = operation_fixture<A>::test(
                    test::splat(dpp::max_value_v<E>), mulop,
                    test::splat(dpp::max_value_v<E>), 2, dpp::max_value_v<E>);
            } else if constexpr (mulop == dpp::nmuladd) {
                dpl::ignore =
                    operation_fixture<A>::test(test::splat(small * small),
                        mulop, test::splat(1 + small), 1 - small, 1);
                // overflow cancels
                dpl::ignore = operation_fixture<A>::test(
                    test::splat(-dpp::max_value_v<E>), mulop,
                    test::splat(dpp::max_value_v<E>), 2, dpp::max_value_v<E>);
            } else if constexpr (mulop == dpp::nmulsub) {
                dpl::ignore =
                    operation_fixture<A>::test(test::splat(small * small),
                        mulop, test::splat(1 + small), 1 - small, -1);
                // overflow cancels
                dpl::ignore = operation_fixture<A>::test(
                    test::splat(-dpp::max_value_v<E>), mulop,
                    test::splat(dpp::max_value_v<E>), 2, -dpp::max_value_v<E>);
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
