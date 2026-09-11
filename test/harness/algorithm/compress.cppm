// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.compress;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class compress {
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using array_t = decltype(test::make_array<E, A>());
    template <typename E>
    using mask_t = dpp::make_canonical_mask_t<E, A>;

    template <dpp::simd_element_for<A> E>
    static constexpr auto expected_op(
        array_t<E> const& val, mask_t<E> mask, array_t<E> const& src) noexcept {
        assert(val.size() == src.size());
        array_t<E> result = val;
        auto const amask = test::to_mask_array<E, A>(mask);
        auto j = 0zu;
        for (auto const i : linear_counter(val)) {
            if (amask[i] != 0) {
                result[j++] = val[i];
            }
        }

        for (; j < abi_traits<E>::size(); ++j) {
            result[j] = src[j];
        }

        return result;
    }

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto const lanes = abi_traits<E>::size();
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
                return compress::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        constexpr auto width = abi_traits<E>::size();
        test::array_generator<A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;

        for (auto const _ : test_count<E>()) {
            auto const val = data_generator(engine);
            auto const mask = mask_generator(engine);

            {
                auto const src = data_generator(engine);
                auto const expected = expected_op<E>(val, mask, src);
                test::operation_fixture<A>::test(
                    test::bitcmp, expected, dpp::compress, val, mask, src);
            }
            {
                auto const azero = [](auto array) {
                    for (auto& val : array) {
                        val = 0;
                    }
                    return array;
                }(val);
                auto const expected = expected_op<E>(val, mask, azero);
                test::operation_fixture<A>::test(test::bitcmp, expected,
                    dpp::compress, val, mask, dpp::zero);
            }
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const val = data_generator(engine);
            auto const src = data_generator(engine);
            auto const azero = [](auto array) {
                for (auto& val : array) {
                    val = 0;
                }
                return array;
            }(src);
            run_const_mask_test<E>([&](auto cmask) {
                {
                    auto const expected = expected_op<E>(val, cmask, src);
                    test::operation_fixture<A>::test(
                        test::bitcmp, expected, dpp::compress, val, cmask, src);
                }
                {
                    auto const expected = expected_op<E>(val, cmask, azero);
                    test::operation_fixture<A>::test(test::bitcmp, expected,
                        dpp::compress, val, cmask, dpp::zero);
                }
            });
        }

        return true;
    }
};
} // namespace dpl::test
