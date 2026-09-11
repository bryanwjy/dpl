// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.lookup;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class lookup {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

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

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return lookup::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        using index_t = dpp::signed_representation_t<E>;
        auto const lanes = dpp::simd_abi_traits<A, E>::size();
        test::array_generator<A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;
        test::array_generator<A, index_t> const idx_generator(0, 2 * lanes);

        for (auto const _ : test_count<E>()) {
            auto const val = data_generator(engine);
            auto const idx = idx_generator(engine);

            {
                auto const src = data_generator(engine);
                auto const expected = [&] {
                    auto expected = val;
                    for (auto const i : linear_counter(expected)) {
                        expected[i] =
                            idx[i] < val.size() ? val[idx[i]] : src[i];
                    }

                    return expected;
                }();
                test::operation_fixture<A>::test(
                    test::bitcmp, expected, dpp::lookup, val, idx, src);
            }
            {
                auto const expected = [&] {
                    auto expected = val;
                    for (auto const i : linear_counter(expected)) {
                        expected[i] = idx[i] < val.size() ? val[idx[i]]
                                                          : static_cast<E>(0);
                    }
                    return expected;
                }();
                test::operation_fixture<A>::test(
                    test::bitcmp, expected, dpp::lookup, val, idx, dpp::zero);
            }
        }

        return true;
    }
};
} // namespace dpl::test
