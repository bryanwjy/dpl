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
    static constexpr bool run(Rng& engine)
    requires dpp::fixed_width_abi<A>
    {
        constexpr auto lanes = abi_traits<E>::size();
        using bitset_t = dpl::bitset<lanes>;
        constexpr auto all_false = bitset_t();
        constexpr auto all_true = ~all_false;

        {
            auto const vall_false = dpp::from_bitset<A, E>(all_false);
            assert(dpp::none_of(vall_false));
            assert(!dpp::all_of(vall_false));
            assert(!dpp::any_of(vall_false));
            assert(!dpp::some_of(vall_false));
        }

        {
            auto const vall_true = dpp::from_bitset<A, E>(all_true);
            assert(!dpp::none_of(vall_true));
            assert(dpp::all_of(vall_true));
            assert(dpp::any_of(vall_true));
            assert(!dpp::some_of(vall_true));
        }

        {
            constexpr test::scalar_generator<size_t> shift_generator(
                1zu, lanes);
            auto const shift = shift_generator(engine);
            auto const some_true = all_true >> shift;
            auto const vsome_true = dpp::from_bitset<A, E>(some_true);
            assert(!dpp::none_of(vsome_true));
            assert(!dpp::all_of(vsome_true));
            assert(dpp::any_of(vsome_true));
            assert(dpp::some_of(vsome_true));
        }

        constexpr test::bit_generator<lanes> bit_generator;
        for (auto i = 0zu; i < lanes; ++i) {
            auto const mask = bit_generator(engine);
            auto const vmask = dpp::from_bitset<A, E>(mask);
            auto const pop = dpl::popcount(mask);
            assert(pop == 0 || dpp::any_of(vmask));
            assert(pop != mask.size() || dpp::all_of(vmask));
            assert(pop == 0 || pop == mask.size() || dpp::some_of(vmask));
            assert(pop != 0 || dpp::none_of(vmask));
        }

        return true;
    }
};
} // namespace dpl::test
