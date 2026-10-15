// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.bitwise;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class bitwise {
    using abi_t = A;
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;

    template <auto Op>
    struct expected_op_t {
        template <typename E>
        static constexpr E operator()(E lhs, E rhs) noexcept
        requires (Op != dpp::bwshift_left && Op != dpp::bwshift_right &&
            Op != dpp::bwnot)
        {
            if constexpr (Op == dpp::bwand) {
                return dpl::bit_cast<E>(dpl::to_bit_representation(lhs) &
                    dpl::to_bit_representation(rhs));
            } else if constexpr (Op == dpp::bwor) {
                return dpl::bit_cast<E>(dpl::to_bit_representation(lhs) |
                    dpl::to_bit_representation(rhs));
            } else if constexpr (Op == dpp::bwandnot) {
                return dpl::bit_cast<E>(dpl::to_bit_representation(lhs) &
                    ~dpl::to_bit_representation(rhs));
            } else if constexpr (Op == dpp::bwornot) {
                return dpl::bit_cast<E>(dpl::to_bit_representation(lhs) |
                    ~dpl::to_bit_representation(rhs));
            } else {
                static_assert(Op == dpp::bwxor);
                return dpl::bit_cast<E>(dpl::to_bit_representation(lhs) ^
                    dpl::to_bit_representation(rhs));
            }
        }

        template <typename E>
        static constexpr E operator()(E arg) noexcept
        requires (Op == dpp::bwnot)
        {
            return dpl::bit_cast<E>(~dpl::to_bit_representation(arg));
        }

        template <typename E>
        static constexpr E operator()(E lhs, size_t rhs) noexcept
        requires (Op == dpp::bwshift_left)
        {
            return dpl::bit_cast<E>(dpl::to_bit_representation(lhs) << rhs);
        }

        template <typename E>
        static constexpr E operator()(E lhs, size_t rhs) noexcept
        requires (Op == dpp::bwshift_right)
        {
            if constexpr (dpl::integral<E>) {
                return static_cast<E>(lhs >> rhs);
            } else {
                return dpl::bit_cast<E>(dpl::to_bit_representation(lhs) >> rhs);
            }
        }
    };

    template <auto bwop>
    static constexpr expected_op_t<bwop> expected_op{};

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
    template <dpp::simd_primitive_operation auto bwop, rng_like Rng,
        dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return bitwise::template run<E, bwop>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E,
        dpp::simd_primitive_operation auto bwop, rng_like Rng>
    requires (bwop != dpp::bwshift_left && bwop != dpp::bwshift_right &&
        bwop != dpp::bwnot)
    static constexpr bool run(Rng& engine) {
        test::array_generator<abi_t, E> const data_generator(
            test::sqrt_range<E>);
        test::mask_generator<A, E> const mask_generator;
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);

        for (auto _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const src = data_generator(engine);

            auto expected = lhs;
            for (auto const i : linear_counter(expected)) {
                expected[i] = expected_op<bwop>(lhs[i], rhs[i]);
            }

            operation_fixture<A>::test(test::bitcmp, expected, bwop, lhs, rhs);
            operation_fixture<A>::test_masked(bwop, src, vtrue, lhs, rhs);
            operation_fixture<A>::test_masked(bwop, src, vfalse, lhs, rhs);
            operation_fixture<A>::test_masked(
                bwop, src, mask_generator(engine), lhs, rhs);
            operation_fixture<A>::test_masked(
                bwop, dpp::zero, mask_generator(engine), lhs, rhs);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const src = data_generator(engine);

            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(bwop, src, cmask, lhs, rhs);
                operation_fixture<A>::test_masked(
                    bwop, dpp::zero, cmask, lhs, rhs);
            });
        }

        return true;
    }

    template <dpp::simd_element_for<A> E,
        dpp::simd_primitive_operation auto bwop, rng_like Rng>
    requires (bwop == dpp::bwshift_left || bwop == dpp::bwshift_right)
    static constexpr bool run(Rng& engine) {
        using shift_t = dpp::unsigned_representation_t<E>;
        test::array_generator<A, E> const data_generator;
        test::array_generator<A, shift_t> const shift_generator(
            0, dpl::type_bit_v<E>);
        test::mask_generator<A, E> const mask_generator;

        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);
        auto const lhs = data_generator(engine);
        auto const rhs = shift_generator(engine);
        auto const src = data_generator(engine);

        for (auto _ : test_count<E>()) {
            auto expected = lhs;
            for (auto const i : linear_counter(expected)) {
                expected[i] = expected_op<bwop>(lhs[i], rhs[i]);
            }

            operation_fixture<A>::test(test::bitcmp, expected, bwop, lhs, rhs);
            operation_fixture<A>::test_masked(bwop, src, vtrue, lhs, rhs);
            operation_fixture<A>::test_masked(bwop, src, vfalse, lhs, rhs);
            operation_fixture<A>::test_masked(
                bwop, src, mask_generator(engine), lhs, rhs);
            operation_fixture<A>::test_masked(
                bwop, dpp::zero, mask_generator(engine), lhs, rhs);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(bwop, src, cmask, lhs, rhs);
                operation_fixture<A>::test_masked(
                    bwop, dpp::zero, cmask, lhs, rhs);
            });
        }

        [&]<size_t I = 0zu, serialized_mt19937 S = {}>(this auto self,
            dpp::immediate<I> = dpp::imm<I>, mt19937_type<S> = {}) {
            if constexpr (I < test_count<E>().size()) {
                constexpr test::scalar_generator<size_t> cshift_generator(
                    0zu, dpl::type_bit_v<E>);

                constexpr auto pair = S.generate_with(cshift_generator);
                constexpr auto crhs = dpp::imm<pair.value>;
                auto const vmask = mask_generator(engine);

                auto expected = lhs;
                for (auto const i : linear_counter(expected)) {
                    expected[i] = expected_op<bwop>(lhs[i], crhs);
                }

                test::operation_fixture<A>::test(
                    test::bitcmp, expected, bwop, lhs, crhs);
                test::operation_fixture<A>::test_masked(
                    bwop, src, vmask, lhs, crhs);
                test::operation_fixture<A>::test_masked(
                    bwop, dpp::zero, vmask, lhs, crhs);

                self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
            }
        }
        ();

        for (auto _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const vmask = mask_generator(engine);
            auto const srhs = shift_generator.scalar(engine);
            auto const src = data_generator(engine);

            auto expected = lhs;
            for (auto const j : linear_counter(expected)) {
                expected[j] = expected_op<bwop>(lhs[j], srhs);
            }

            test::operation_fixture<A>::test(
                test::bitcmp, expected, bwop, lhs, srhs);
            test::operation_fixture<A>::test_masked(
                bwop, src, vmask, lhs, srhs);
            test::operation_fixture<A>::test_masked(
                bwop, dpp::zero, vmask, lhs, srhs);
        }

        return true;
    }

    template <dpp::simd_element_for<A> E,
        dpp::simd_primitive_operation auto bwop, rng_like Rng>
    requires (bwop == dpp::bwnot)
    static constexpr bool run(Rng& engine) {
        test::array_generator<abi_t, E> const data_generator(
            test::sqrt_range<E>);
        test::mask_generator<A, E> const mask_generator;
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);

        for (auto _ : test_count<E>()) {
            auto const arg = data_generator(engine);
            auto const src = data_generator(engine);

            auto expected = arg;
            for (auto& val : expected) {
                val = expected_op<bwop>(val);
            }

            operation_fixture<A>::test(test::bitcmp, expected, bwop, arg);
            operation_fixture<A>::test_masked(bwop, src, vtrue, arg);
            operation_fixture<A>::test_masked(bwop, src, vfalse, arg);
            operation_fixture<A>::test_masked(
                bwop, src, mask_generator(engine), arg);
            operation_fixture<A>::test_masked(
                bwop, dpp::zero, mask_generator(engine), arg);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto count = test_count<E>().size();
            auto const arg = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const src = data_generator(engine);

            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(bwop, src, cmask, arg);
                operation_fixture<A>::test_masked(bwop, dpp::zero, cmask, arg);
            });
        }

        return true;
    }
};
} // namespace dpl::test
