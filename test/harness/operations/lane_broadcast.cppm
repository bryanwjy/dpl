// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.lane_broadcast;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class lane_broadcast {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = abi_traits<E>::size();
        auto const limit = []() {
            if consteval {
                return 4zu;
            } else {
                return abi_traits<E>::size();
            }
        }();

        return linear_counter(
            lanes < limit ? static_cast<size_t>(lanes) : limit);
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
                return lane_broadcast::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<A, E> const data_generator;
        test::scalar_generator<size_t> const idx_generator(
            0zu, abi_traits<E>::size());
        test::mask_generator<A, E> const mask_generator;

        auto const vtrue = dpp::broadcast<A, E>(true);
        auto const vfalse = dpp::broadcast<A, E>(false);

        for (auto const _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const src = data_generator(engine);
            auto const idx = idx_generator(engine);
            auto expected = lhs;
            for (auto& val : expected) {
                val = lhs[idx];
            }

            test::operation_fixture<A>::test(
                test::bitcmp, expected, dpp::broadcast_lane, lhs, idx);
            auto const mask = mask_generator(engine);
            test::operation_fixture<A>::test_masked(
                dpp::broadcast_lane, src, mask, lhs, idx);
            test::operation_fixture<A>::test_masked(
                dpp::broadcast_lane, src, vtrue, lhs, idx);
            test::operation_fixture<A>::test_masked(
                dpp::broadcast_lane, src, vfalse, lhs, idx);
            test::operation_fixture<A>::test_masked(
                dpp::broadcast_lane, dpp::zero, vfalse, lhs, idx);
        }

        run_const_mask_test<E>([&](auto cmask) {
            auto const lhs = data_generator(engine);
            auto const idx = idx_generator(engine);
            auto const src = data_generator(engine);
            test::operation_fixture<A>::test_masked(
                dpp::broadcast_lane, src, cmask, lhs, idx);
            test::operation_fixture<A>::test_masked(
                dpp::broadcast_lane, dpp::zero, cmask, lhs, idx);
        });

        // OOB is implementation-defined, not tested

        // There is no way to ensure index bounds for scalable ABIs
        // they should typically fallback to runtime index anyway
        // so only fixed-width ABIs are tested for immediate indices
        if constexpr (dpp::fixed_width_abi<A>) {
            [&]<size_t I = 0zu, serialized_mt19937 S = {}>(this auto self,
                dpp::immediate<I> = dpp::imm<I>, mt19937_type<S> = {}) {
                if constexpr (I < test_count<E>().size()) {
                    constexpr test::scalar_generator<size_t> cshift_generator(
                        0zu, abi_traits<E>::size());

                    constexpr auto pair = S.generate_with(cshift_generator);
                    constexpr auto cidx = dpp::imm<pair.value>;

                    auto const lhs = data_generator(engine);
                    auto const src = data_generator(engine);
                    auto expected = lhs;
                    for (auto& val : expected) {
                        val = lhs[cidx];
                    }

                    test::operation_fixture<A>::test(
                        test::bitcmp, expected, dpp::broadcast_lane, lhs, cidx);
                    auto const mask = mask_generator(engine);
                    test::operation_fixture<A>::test_masked(
                        dpp::broadcast_lane, src, mask, lhs, cidx);
                    test::operation_fixture<A>::test_masked(
                        dpp::broadcast_lane, src, vtrue, lhs, cidx);
                    test::operation_fixture<A>::test_masked(
                        dpp::broadcast_lane, src, vfalse, lhs, cidx);
                    test::operation_fixture<A>::test_masked(
                        dpp::broadcast_lane, dpp::zero, vfalse, lhs, cidx);

                    self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
                }
            }
            ();
        }

        return true;
    }
};

} // namespace dpl::test
