// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.reinterpretation;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class reinterpretation {
    using abi_t = A;
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return reinterpretation::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<abi_t, dpl::uint64> const data_generator;

        // Go through a few iterations
        for (auto const i : linear_counter(8)) {
            auto const data = data_generator(engine);
            auto const expected = test::reinterpret_array<E>(data);
            operation_fixture<A>::test(
                test::bitcmp, expected, dpp::reinterpret<E>, data);
        }

        return true;
    }
};

} // namespace dpl::test
