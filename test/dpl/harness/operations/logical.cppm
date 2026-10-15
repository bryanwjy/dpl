// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.logical;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class logical {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return logical::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        auto const lanes = abi_traits<E>::size();
        auto const mall_false = dpp::broadcast<A, E>(false);
        auto const mall_true = dpp::broadcast<A, E>(true);

        {
            assert(dpp::none_of(mall_false));
            assert(!dpp::all_of(mall_false));
            assert(!dpp::any_of(mall_false));
            assert(!dpp::some_of(mall_false));
        }

        {
            assert(!dpp::none_of(mall_true));
            assert(dpp::all_of(mall_true));
            assert(dpp::any_of(mall_true));
            assert(!dpp::some_of(mall_true));
        }

        auto const vzero = dpp::broadcast<A, E>(dpp::zero);
        for (test::mask_generator<A, E> const mask_generator;
            auto const _ : linear_counter([]() {
                if consteval {
                    return 4zu < abi_traits<E>::size() ? 4zu
                                                       : abi_traits<E>::size();
                } else {
                    return abi_traits<E>::size();
                }
            }())) {
            auto const mask = mask_generator(engine);

            auto const expected = [&]() {
                auto const vmask = dpp::select(mask, vzero, dpp::all_bits);
                dynamic_array<E> amask(lanes);
                dpp::store(vmask, amask.data());
                auto count = 0zu;
                for (auto const val : amask) {
                    count += val == 0;
                }

                return dpl::bitset<4>(count == 0, count == lanes, count != 0,
                    count - 1 < lanes - 1);
                // for some_of: if count == 0 -> overflow to size_t max
            }();
            assert(dpp::none_of(mask) == expected[0]);
            assert(dpp::all_of(mask) == expected[1]);
            assert(dpp::any_of(mask) == expected[2]);
            assert(dpp::some_of(mask) == expected[3]);
        }

        return true;
    }
};
} // namespace dpl::test
