// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.arithmetic;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_primitive_operation auto arop, dpp::simd_abi A>
class arithmetic {
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept
    requires (arop == dpp::add)
    {
        if constexpr (dpl::signed_integral<E>) {
            return dpl::to_signed(
                dpl::to_unsigned(lhs) + dpl::to_unsigned(rhs));
        } else {
            return lhs + rhs;
        }
    }

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept
    requires (arop == dpp::subtract)
    {
        if constexpr (dpl::signed_integral<E>) {
            return dpl::to_signed(
                dpl::to_unsigned(lhs) - dpl::to_unsigned(rhs));
        } else {
            return lhs - rhs;
        }
    }

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept
    requires (arop == dpp::multiply)
    {
        return lhs * rhs;
    }

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept
    requires (arop == dpp::divide)
    {
        return lhs / rhs;
    }

    template <typename E, rng_like Rng>
    static constexpr void operation_specific_tests(
        array_generator<A, E> const& generator, Rng& engine)
    requires (arop == dpp::add)
    {
        if constexpr (dpl::integral<E>) {
            operation_fixture<A>::test(test::splat(dpp::zero_v<E>), dpp::add,
                test::splat(dpp::zero_v<E>), test::splat(dpp::zero_v<E>));
            auto const rnd = generator(engine);
            operation_fixture<A>::test(rnd, dpp::add, dpp::zero, rnd);
            operation_fixture<A>::test(rnd, dpp::add, rnd, dpp::zero);

            constexpr auto max = dpl::integral_traits<E>::max_value;
            constexpr auto min = dpl::integral_traits<E>::min_value;

            auto const wraparound_test = []() {
                if constexpr (dpl::unsigned_integral<E>) {
                    return true;
                } else if not consteval {
                    return true;
                } else {
                    return false;
                }
            }();

            if (wraparound_test) {
                operation_fixture<A>::test(test::splat(min), dpp::add,
                    test::splat(max), test::splat<E>(1));
                operation_fixture<A>::test(test::splat(max), dpp::add,
                    test::splat(min), test::splat<E>(-1));
                operation_fixture<A>::test(test::splat<E>(max - 1), dpp::add,
                    test::splat(max), test::splat(max));
                operation_fixture<A>::test(test::splat<E>(-1), dpp::add,
                    test::splat(min), test::splat(max));
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::add, test::splat<E>(-0.0), 0.0);
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::add, test::splat<E>(0.0), 0.0);
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::add, test::splat<E>(-0.0), -0.0);
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::add, test::splat<E>(1.0), -1.0);

            if (!test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;
                operation_fixture<A>::test(
                    test::splat<E>(inf), dpp::add, test::splat<E>(inf), 1.0);
                operation_fixture<A>::test(
                    test::splat<E>(-inf), dpp::add, test::splat<E>(-inf), -1.0);

                if not consteval {
                    // nan producing arithmetic is not allowed at constexpr
                    operation_fixture<A>::test(test::nancmp,
                        test::splat<E>(dpp::nan), dpp::add, test::splat<E>(inf),
                        -inf);
                    operation_fixture<A>::test(test::nancmp,
                        test::splat<E>(dpp::nan), dpp::add,
                        test::splat<E>(dpp::nan), 1.0);
                    operation_fixture<A>::test(test::nancmp,
                        test::splat<E>(dpp::nan), dpp::add, 1.0,
                        test::splat<E>(dpp::nan));
                }
            }
        }
    }

    template <typename E, rng_like Rng>
    static constexpr void operation_specific_tests(
        array_generator<A, E> const& generator, Rng& engine)
    requires (arop == dpp::subtract)
    {
        if constexpr (dpl::integral<E>) {
            operation_fixture<A>::test(
                test::splat<E>(0), dpp::subtract, test::splat<E>(0), 0);
            auto const rnd = generator(engine);
            operation_fixture<A>::test(rnd, dpp::subtract, rnd, 0);
            operation_fixture<A>::test(
                test::splat<E>(-rnd[0]), dpp::subtract, 0, test::splat(rnd[0]));
            operation_fixture<A>::test(
                test::splat<E>(0), dpp::subtract, rnd, rnd);

            constexpr auto max = dpl::integral_traits<E>::max_value;
            constexpr auto min = dpl::integral_traits<E>::min_value;

            auto const wraparound_test = []() {
                if constexpr (dpl::unsigned_integral<E>) {
                    return true;
                } else if not consteval {
                    return true;
                } else {
                    return false;
                }
            }();
            if (wraparound_test) {
                operation_fixture<A>::test(test::splat(max), dpp::subtract,
                    test::splat(min), test::splat<E>(1));
                operation_fixture<A>::test(test::splat(min), dpp::subtract,
                    test::splat(max), test::splat<E>(-1));
                operation_fixture<A>::test(test::splat<E>(1), dpp::subtract,
                    test::splat(min), test::splat(max));
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::subtract, test::splat<E>(-0.0), 0.0);
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::subtract, test::splat<E>(0.0), 0.0);
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::subtract, test::splat<E>(-0.0), -0.0);
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::subtract, test::splat<E>(1.0), 1.0);
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::subtract, test::splat<E>(-1.0), -1.0);

            if (!dpl::test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;
                operation_fixture<A>::test(
                    test::splat(inf), dpp::subtract, test::splat(inf), 1.0);
                operation_fixture<A>::test(
                    test::splat(-inf), dpp::subtract, test::splat(-inf), -1.0);
                if not consteval {
                    // nan producing arithmetic is not allowed at constexpr
                    operation_fixture<A>::test(test::nancmp,
                        test::splat<E>(dpp::nan), dpp::subtract,
                        test::splat<E>(-inf), -inf);
                    operation_fixture<A>::test(test::nancmp,
                        test::splat<E>(dpp::nan), dpp::subtract,
                        test::splat<E>(inf), inf);
                    operation_fixture<A>::test(test::nancmp,
                        test::splat<E>(dpp::nan), dpp::subtract,
                        test::splat<E>(dpp::nan), 1.0);
                    operation_fixture<A>::test(test::nancmp,
                        test::splat<E>(dpp::nan), dpp::subtract, 1.0,
                        test::splat<E>(dpp::nan));
                }
            }
        }
    }

    template <typename E, rng_like Rng>
    static constexpr void operation_specific_tests(
        array_generator<A, E> const& generator, Rng& engine)
    requires (arop == dpp::multiply)
    {
        auto const rnd = generator(engine);
        operation_fixture<A>::test(rnd, dpp::multiply, rnd, 1);
        operation_fixture<A>::test(rnd, dpp::multiply, 1, rnd);
        operation_fixture<A>::test(test::splat<E>(0), dpp::multiply, 0, rnd);
        operation_fixture<A>::test(test::splat<E>(0), dpp::multiply, rnd, 0);

        if constexpr (dpl::integral<E>) {
            if not consteval {
                constexpr auto int_min = dpl::integral_traits<E>::min_value;
                constexpr auto int_max = dpl::integral_traits<E>::max_value;
                operation_fixture<A>::test(
                    test::splat<E>(-2), dpp::multiply, test::splat(int_max), 2);

                if constexpr (dpl::signed_integral<E>) {
                    operation_fixture<A>::test(test::splat<E>(0), dpp::multiply,
                        test::splat(int_min), 2);
                }

                operation_fixture<A>::test(test::splat<E>(1), dpp::multiply,
                    test::splat(int_max), int_max);
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::multiply, test::splat<E>(-0.0), 1.0);
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::multiply, test::splat<E>(0.0), 1.0);
            operation_fixture<A>::test(
                test::splat<E>(0.0), dpp::multiply, test::splat<E>(-0.0), -0.0);
            operation_fixture<A>::test(
                test::splat<E>(1.0), dpp::multiply, test::splat<E>(-1.0), -1.0);

            if (test::ieee_denormal()) {
                using ubit_t = dpp::unsigned_representation_t<E>;
                auto const denorm = dpl::bit_cast<E>(ubit_t(1));
                operation_fixture<A>::test(test::splat<E>(0), dpp::multiply,
                    test::splat(denorm), denorm);
                operation_fixture<A>::test(
                    test::splat(dpl::bit_cast<E>(ubit_t(2))), dpp::multiply,
                    test::splat(denorm), 2.0);
            }

            if (!test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;
                operation_fixture<A>::test(
                    test::splat(inf), dpp::multiply, test::splat(inf), 2.0);
                operation_fixture<A>::test(
                    test::splat(-inf), dpp::multiply, test::splat(inf), -2.0);
                operation_fixture<A>::test(
                    test::splat(-inf), dpp::multiply, test::splat(-inf), 2.0);
                operation_fixture<A>::test(
                    test::splat(inf), dpp::multiply, test::splat(-inf), -2.0);

                operation_fixture<A>::test(
                    test::splat(inf), dpp::multiply, test::splat(inf), inf);
                operation_fixture<A>::test(
                    test::splat(inf), dpp::multiply, test::splat(-inf), -inf);
                operation_fixture<A>::test(
                    test::splat(inf), dpp::multiply, test::splat(-inf), -inf);

                operation_fixture<A>::test(test::splat(inf), dpp::multiply,
                    test::splat<E>(dpp::max_value), dpp::max_value_v<E>);
                operation_fixture<A>::test(test::splat(-inf), dpp::multiply,
                    test::splat<E>(dpp::max_value), -dpp::max_value_v<E>);
            }
        }
    }

    template <typename E, rng_like Rng>
    static constexpr void operation_specific_tests(
        array_generator<A, E> const& generator, Rng& engine)
    requires (arop == dpp::divide)
    {
        auto const rnd = generator(engine);
        operation_fixture<A>::test(test::splat<E>(0), dpp::divide, 0, rnd);
        operation_fixture<A>::test(rnd, dpp::divide, rnd, 1);
        operation_fixture<A>::test(
            test::splat<E>(-rnd[0]), dpp::divide, test::splat(rnd[0]), -1);

        if (test::ieee_denormal()) {
            using ubit_t = dpp::unsigned_representation_t<E>;
            auto const denorm = dpl::bit_cast<E>(ubit_t(1));
            operation_fixture<A>::test(test::splat<E>(dpp::infinity),
                dpp::divide, 1, test::splat(denorm));
            operation_fixture<A>::test(test::splat(denorm), dpp::divide,
                test::splat(dpl::bit_cast<E>(ubit_t(2))), 2.0);
        }

        if (!test::finite_math_only()) {
            constexpr auto inf = dpp::infinity_v<E>;

            operation_fixture<A>::test(
                test::splat(inf), dpp::divide, test::splat(inf), 2.0);
            operation_fixture<A>::test(
                test::splat(-inf), dpp::divide, test::splat(inf), -2.0);
            operation_fixture<A>::test(
                test::splat(-inf), dpp::divide, test::splat(-inf), 2.0);
            operation_fixture<A>::test(
                test::splat(inf), dpp::divide, test::splat(-inf), -2.0);

            operation_fixture<A>::test(test::bitcmp, test::splat<E>(0.0),
                dpp::divide, 2.0, test::splat(inf));
            operation_fixture<A>::test(test::bitcmp, test::splat<E>(-0.0),
                dpp::divide, 2.0, test::splat(-inf));
            operation_fixture<A>::test(test::bitcmp, test::splat<E>(-0.0),
                dpp::divide, -2.0, test::splat(inf));
            operation_fixture<A>::test(test::bitcmp, test::splat<E>(0.0),
                dpp::divide, -2.0, test::splat(-inf));

            if not consteval {
                operation_fixture<A>::test(
                    test::splat(inf), dpp::divide, 1, test::splat<E>(0));
                operation_fixture<A>::test(
                    test::splat(-inf), dpp::divide, -1, test::splat<E>(0));

                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide, test::splat(inf),
                    inf);
                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide, test::splat(inf),
                    -inf);
                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide, test::splat(-inf),
                    inf);
                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide, test::splat(-inf),
                    -inf);

                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide, test::splat<E>(0.0),
                    0.0);
                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide, test::splat<E>(-0.0),
                    0.0);
                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide, test::splat<E>(0.0),
                    -0.0);
                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide, test::splat<E>(-0.0),
                    -0.0);

                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide,
                    test::splat<E>(dpp::nan), 1);
                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide, 1,
                    test::splat<E>(dpp::nan));
                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide, inf,
                    test::splat<E>(dpp::nan));
                operation_fixture<A>::test(test::nancmp,
                    test::splat<E>(dpp::nan), dpp::divide,
                    test::splat<E>(dpp::nan), inf);
            }
        }
    }

    template <typename E>
    static constexpr auto mixmask() noexcept
    requires (arop == dpp::addsub)
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
    requires (arop == dpp::subadd)
    {
        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto lanes = abi_traits<E>::size();
            using bitset_t = dpl::bitset<lanes>;
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
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return arithmetic::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        array_generator<A, E> const generator(test::sqrt_range<E>);
        mask_generator<A, E> const mask_generator;
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);

        for (auto const _ : test_count<E>()) {
            auto const lhs = generator(engine);
            auto const rhs = generator(engine);
            auto const src = generator(engine);
            auto expected = lhs;
            for (auto const i : linear_counter(expected)) {
                expected[i] = expected_op(lhs[i], rhs[i]);
            }

            operation_fixture<A>::test(test::bitcmp, expected, arop, lhs, rhs);
            operation_fixture<A>::test_masked(arop, src, vtrue, lhs, rhs);
            operation_fixture<A>::test_masked(arop, src, vfalse, lhs, rhs);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(arop, src, mask, lhs, rhs);
            operation_fixture<A>::test_masked(arop, dpp::zero, mask, lhs, rhs);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const lhs = generator(engine);
            auto const rhs = generator(engine);
            auto const src = generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(arop, src, cmask, lhs, rhs);
                operation_fixture<A>::test_masked(
                    arop, dpp::zero, cmask, lhs, rhs);
            });
        }

        operation_specific_tests<E>(generator, engine);

        return true;
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires (arop == dpp::addsub || arop == dpp::subadd)
    {
        array_generator<A, E> const generator(test::sqrt_range<E>);
        mask_generator<A, E> const mask_generator;
        constexpr auto expected_op = [](vec_t<E> vlhs, vec_t<E> vrhs) noexcept {
            return dpp::select(
                mixmask<E>(), dpp::add(vlhs, vrhs), dpp::subtract(vlhs, vrhs));
        };

        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);
        for (auto const _ : test_count<E>()) {
            auto const lhs = generator(engine);
            auto const rhs = generator(engine);
            auto const vlhs = dpp::load<E, A>(lhs.data());
            auto const vrhs = dpp::load<E, A>(rhs.data());
            auto const vexpected = expected_op(vlhs, vrhs);
            auto const vactual = arop(vlhs, vrhs);

            operation_fixture<A>::test(vexpected, arop, vlhs, vrhs);

            auto const src = generator(engine);
            auto const vsrc = dpp::load<E, A>(src.data());

            operation_fixture<A>::test_masked(arop, vsrc, vtrue, vlhs, vrhs);
            operation_fixture<A>::test_masked(arop, vsrc, vfalse, vlhs, vrhs);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(arop, vsrc, mask, vlhs, vrhs);
            operation_fixture<A>::test_masked(
                arop, dpp::zero, mask, vlhs, vrhs);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const lhs = generator(engine);
            auto const rhs = generator(engine);
            auto const src = generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(arop, src, cmask, lhs, rhs);
                operation_fixture<A>::test_masked(
                    arop, dpp::zero, cmask, lhs, rhs);
            });
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using addition = arithmetic<dpp::add, A>;
export template <dpp::simd_abi A>
using subtraction = arithmetic<dpp::subtract, A>;
export template <dpp::simd_abi A>
using multiplication = arithmetic<dpp::multiply, A>;
export template <dpp::simd_abi A>
using division = arithmetic<dpp::divide, A>;
export template <dpp::simd_abi A>
using addsubtraction = arithmetic<dpp::addsub, A>;
export template <dpp::simd_abi A>
using subaddition = arithmetic<dpp::subadd, A>;
} // namespace dpl::test
