// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.byteswap;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class byteswap {
private:
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
                return byteswap::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
        test::array_generator<A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);

        for (auto const _ : test_count<E>()) {
            auto const data = data_generator(engine);
            auto const src = data_generator(engine);
            auto const expected = [&]() {
                auto result = data;
                for (auto const i : linear_counter(result)) {
                    result[i] = dpl::byteswap(data[i]);
                }
                return result;
            }();

            test::operation_fixture<A>::test(expected, dpp::byteswap, data);
            test::operation_fixture<A>::test_masked(
                dpp::byteswap, src, vtrue, data);
            test::operation_fixture<A>::test_masked(
                dpp::byteswap, src, vfalse, data);
            auto const mask = mask_generator(engine);
            test::operation_fixture<A>::test_masked(
                dpp::byteswap, src, mask, data);
            test::operation_fixture<A>::test_masked(
                dpp::byteswap, dpp::zero, mask, data);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const data = data_generator(engine);
            auto const src = data_generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(
                    dpp::byteswap, src, cmask, data);
                operation_fixture<A>::test_masked(
                    dpp::byteswap, dpp::zero, cmask, data);
            });
        }

        return true;
    }
};

} // namespace dpl::test
