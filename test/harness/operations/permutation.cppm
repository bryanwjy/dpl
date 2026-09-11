// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.permutation;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class permutation {

    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto const lanes = abi_traits<E>::size();
        auto const limit = []() {
            if consteval {
                return 2zu;
            } else {
                return 7zu;
            }
        }();

        auto const shift =
            dpp::min(static_cast<size_t>(dpl::bit_width(lanes)), limit);
        return linear_counter(1zu << shift);
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
                return permutation::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;
        using index_t = dpp::signed_representation_t<E>;

        auto const lanes = dpp::simd_abi_traits<A, E>::size();
        test::array_generator<A, index_t> const idx_generator(0, lanes);

        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);
        for (auto const _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const idx = idx_generator(engine);
            auto const src = data_generator(engine);

            auto expected = lhs;
            for (auto const i : linear_counter(expected)) {
                expected[i] = lhs[idx[i]];
            }

            test::operation_fixture<A>::test(
                test::bitcmp, expected, dpp::permute, lhs, idx);
            test::operation_fixture<A>::test_masked(
                dpp::permute, src, vtrue, lhs, idx);
            test::operation_fixture<A>::test_masked(
                dpp::permute, src, vfalse, lhs, idx);
            auto const mask = mask_generator(engine);
            test::operation_fixture<A>::test_masked(
                dpp::permute, src, mask, lhs, idx);
            test::operation_fixture<A>::test_masked(
                dpp::permute, dpp::zero, mask, lhs, idx);
        }

        {
            auto const lhs = data_generator(engine);
            auto const vidx = dpp::lane_index<E, A>();
            test::operation_fixture<A>::test(
                test::bitcmp, lhs, dpp::permute, lhs, vidx);
        }

        for (auto const _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const sidx = idx_generator.scalar(engine);
            auto const vidx = dpp::broadcast<A>(sidx);
            auto const vexpected = dpp::broadcast<E, A>(lhs[sidx]);
            test::operation_fixture<A>::test(
                test::bitcmp, vexpected, dpp::permute, lhs, vidx);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            run_const_mask_test<E>([&](auto cmask) {
                auto const lhs = data_generator(engine);
                auto const idx = idx_generator(engine);
                auto const src = data_generator(engine);

                auto expected = lhs;
                for (auto const i : linear_counter(expected)) {
                    expected[i] = lhs[idx[i]];
                }

                operation_fixture<A>::test_masked(
                    dpp::permute, src, cmask, lhs, idx);
                operation_fixture<A>::test_masked(
                    dpp::permute, dpp::zero, cmask, lhs, idx);
            });

            constexpr test::array_generator<A, index_t> cshift_generator(
                0zu, abi_traits<E>::size());
            auto const lhs = data_generator(engine);
            auto const src = data_generator(engine);
            [&]<size_t I = 0zu, serialized_mt19937 S = {}>(this auto self,
                dpp::immediate<I> = dpp::imm<I>, mt19937_type<S> = {}) {
                if constexpr (I < test_count<E>().size()) {
                    constexpr auto pair = S.generate_with(cshift_generator);
                    constexpr auto cidx = [&]<size_t... Is>(
                                              dpl::index_sequence<Is...>) {
                        return dpl::index_sequence<pair.value[Is]...>{};
                    }(dpl::make_index_sequence<abi_traits<E>::size()>{});

                    auto expected = lhs;
                    for (auto const i : linear_counter(expected)) {
                        expected[i] = lhs[pair.value[i]];
                    }

                    test::operation_fixture<A>::test(
                        test::bitcmp, expected, dpp::permute, lhs, cidx);
                    test::operation_fixture<A>::test_masked(
                        dpp::permute, src, vtrue, lhs, cidx);
                    test::operation_fixture<A>::test_masked(
                        dpp::permute, src, vfalse, lhs, cidx);
                    auto const mask = mask_generator(engine);
                    test::operation_fixture<A>::test_masked(
                        dpp::permute, src, mask, lhs, cidx);
                    test::operation_fixture<A>::test_masked(
                        dpp::permute, dpp::zero, mask, lhs, cidx);

                    self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
                }
            }
            ();
        }

        return true;
    }
};
} // namespace dpl::test
