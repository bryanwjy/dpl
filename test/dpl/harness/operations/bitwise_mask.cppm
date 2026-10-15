// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.bitwise_mask;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class bitwise_mask {
    using abi_t = A;
    template <typename E>
    using mask_t = dpp::make_canonical_mask_t<E, A>;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<abi_t, E>;

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto const lanes = dpp::simd_abi_traits<A, E>::size();
        if consteval {
            constexpr auto limit = 4zu;
            return linear_counter(lanes < limit ? lanes : limit);
        } else {
            return linear_counter(dpp::simd_abi_traits<A, E>::size());
        }
    }

    template <typename E>
    static constexpr bool demorgans(mask_t<E> lhs, mask_t<E> rhs) noexcept {
        return dpp::all_of(dpp::cmpeq(dpp::bwnot(dpp::bwand(lhs, rhs)),
            dpp::bwor(dpp::bwnot(lhs), dpp::bwnot(rhs))));
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return bitwise_mask::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        mask_generator<A, E> const generator;

        // tests bwnot, bwand, bwor
        for (auto const _ : test_count<E>()) {
            auto const lhs = generator(engine);
            auto const rhs = generator(engine);
            assert(demorgans<E>(lhs, rhs));
        }

        for (auto const _ : test_count<E>()) {
            auto const lhs = generator(engine);
            auto const rhs = generator(engine);
            assert(dpp::all_of(dpp::cmpeq(
                dpp::bwandnot(lhs, rhs), dpp::bwand(lhs, dpp::bwnot(rhs)))));
            assert(dpp::all_of(dpp::cmpeq(
                dpp::bwornot(lhs, rhs), dpp::bwor(lhs, dpp::bwnot(rhs)))));
        }

        auto const vnone = dpp::broadcast<E, A>(false);
        for (auto const _ : test_count<E>()) {
            auto const lhs = generator(engine);
            auto const rhs = generator(engine);
            assert(dpp::none_of(dpp::bwxor(lhs, lhs)));
            assert(dpp::all_of(
                dpp::cmpeq(dpp::bwxor(dpp::bwxor(lhs, rhs), rhs), lhs)));
            assert(dpp::all_of(dpp::cmpeq(dpp::bwxor(lhs, vnone), lhs)));
        }

        scalar_generator<size_t> const shift_generator(
            0zu, abi_traits<E>::size());
        if constexpr (dpp::fixed_width_abi<A>) {
            auto const lanes = abi_traits<E>::size();
            for (auto const _ : test_count<E>()) {
                auto const lhs = generator(engine);
                auto const lbits = dpp::to_bitset(lhs);
                auto const shift = shift_generator(engine);

                assert(dpp::to_bitset(dpp::bwshift_left(lhs, shift)) ==
                    (lbits << shift));
                assert(dpp::to_bitset(dpp::bwshift_right(lhs, shift)) ==
                    (lbits >> shift));
            }
        } else {
            using sint_t = dpp::signed_representation_t<E>;
            auto const vzero = dpp::broadcast<E, A>(dpp::zero);
            for (auto const _ : test_count<E>()) {
                auto const lhs = generator(engine);
                auto const vlhs = dpp::select(lhs, dpp::all_bits, vzero);
                auto const shift = shift_generator(engine);

                {
                    auto const shifted = dpp::bwshift_left(lhs, shift);
                    auto const vshifted =
                        dpp::select(shifted, dpp::all_bits, vzero);
                    auto const vexpected = [&]() {
                        dynamic_array<sint_t> alhs(abi_traits<E>::size());
                        dynamic_array<sint_t> ashifted(abi_traits<E>::size());
                        dpp::store(vlhs, alhs.data());
                        for (auto const i : linear_counter(alhs)) {
                            ashifted[i] =
                                i < shift ? dpp::zero_v<E> : alhs[i - shift];
                        }
                        return dpp::load<A>(ashifted.data());
                    }();

                    assert(dpp::all_of(dpp::cmpeq(vexpected, vshifted)));
                }
                {
                    auto const shifted = dpp::bwshift_right(lhs, shift);
                    auto const vshifted =
                        dpp::select(shifted, dpp::all_bits, vzero);
                    auto const vexpected = [&]() {
                        dynamic_array<sint_t> alhs(abi_traits<E>::size());
                        dynamic_array<sint_t> ashifted(abi_traits<E>::size());
                        dpp::store(vlhs, alhs.data());
                        for (auto const i : linear_counter(alhs)) {
                            ashifted[i] = i + shift < alhs.size()
                                ? alhs[i + shift]
                                : dpp::zero_v<E>;
                        }

                        return dpp::load<A>(ashifted.data());
                    }();
                    assert(dpp::all_of(dpp::cmpeq(vexpected, vshifted)));
                }
            }
        }

        return true;
    }
};
} // namespace dpl::test
