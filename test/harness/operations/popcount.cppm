// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.popcount;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class popcount {
    using abi_t = A;
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using array_t = conditional_t<dpp::fixed_width_abi<A>,
        test::array<E, abi_traits<E>::size>, test::dynamic_array<E>>;

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
                return popcount::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        // Go through a few iterations
        using count_t = dpp::unsigned_representation_t<E>;
        test::array_generator<A, E> const data_generator;
        test::array_generator<A, count_t> const src_generator(
            dpl::type_bit_v<E>, dpl::integral_traits<count_t>::max_value);
        mask_generator<A, E> const mask_generator;
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);

        for (auto const _ : test_count<E>()) {
            auto const data = data_generator(engine);
            auto const src = src_generator(engine);
            auto const expected = [&]() {
                auto result = src;
                for (auto const i : linear_counter(result)) {
                    result[i] = dpl::popcount(dpl::to_unsigned(data[i]));
                }
                return result;
            }();

            test::operation_fixture<A>::test(expected, dpp::popcount, data);
            operation_fixture<A>::test_masked(dpp::popcount, src, vtrue, data);
            operation_fixture<A>::test_masked(dpp::popcount, src, vfalse, data);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(dpp::popcount, src, mask, data);
            operation_fixture<A>::test_masked(
                dpp::popcount, dpp::zero, mask, data);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const data = data_generator(engine);
            auto const src = src_generator(engine);

            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(
                    dpp::popcount, src, cmask, data);
                operation_fixture<A>::test_masked(
                    dpp::popcount, dpp::zero, cmask, data);
            });
        }

        {
            // popcount mask
            for (auto const vzero = dpp::broadcast<A, E>(dpp::zero);
                auto const _ : test_count<E>()) {
                auto const mask = mask_generator(engine);
                auto const expected = [&]() {
                    // zero if true; it is easier to compare, i.e. NANs
                    auto const vmask = dpp::select(mask, vzero, dpp::all_bits);
                    auto amask = test::make_array<E, A>();
                    dpp::store(vmask, amask.data());
                    auto count = 0zu;
                    for (auto const val : amask) {
                        count += val == 0;
                    }

                    return count;
                }();

                assert(expected == dpp::popcount(mask));
            }
        }

        return true;
    }
};

} // namespace dpl::test
