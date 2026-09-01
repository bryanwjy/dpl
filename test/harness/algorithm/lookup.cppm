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
        test::array_generator<A, E> const data_generator(test::half_range);
        using index_t = dpp::signed_representation_t<E>;
        constexpr auto lanes = dpp::simd_abi_traits<A, E>::size();
        test::array_generator<A, index_t> const idx_generator(0, 2 * lanes);
        test::scalar_generator<E> const src_generator(
            dpp::max_value_v<E> / 4 * 3, dpp::max_value_v<E>);

        for (auto i = 0; i < 3; ++i) {
            auto const val = data_generator(engine);
            auto const idx = idx_generator(engine);
            auto const src = src_generator(engine);

            auto expected = val;
            {
                for (auto i = 0zu; i < expected.size(); ++i) {
                    expected[i] = idx[i] < val.size() ? val[idx[i]] : src;
                }

                auto const vval = dpp::load<A>(val.data());
                auto const vidx = dpp::load<A>(idx.data());
                auto const vexpected = dpp::load<A>(expected.data());
                auto const vactual =
                    dpp::lookup(vval, vidx, dpp::broadcast<A, E>(src));
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
            }
        }

        for (auto i = 0; i < 3; ++i) {
            auto const val = data_generator(engine);
            auto const idx = idx_generator(engine);

            auto expected = val;
            {
                for (auto i = 0zu; i < expected.size(); ++i) {
                    expected[i] =
                        idx[i] < val.size() ? val[idx[i]] : static_cast<E>(0);
                }

                auto const vval = dpp::load<A>(val.data());
                auto const vidx = dpp::load<A>(idx.data());
                auto const vexpected = dpp::load<A>(expected.data());
                auto const vactual = dpp::lookup(vval, vidx, dpp::zero);
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
            }
        }

        return true;
    }
};
} // namespace dpl::test
