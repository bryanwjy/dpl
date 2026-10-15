// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.bitrot;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_primitive_operation auto rotop, dpp::simd_abi A>
class bitrot {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr E expected_op(E lhs, size_t rhs) noexcept
    requires (rotop == dpp::rotr)
    {
        return static_cast<E>(dpl::rotr(dpl::to_unsigned(lhs), rhs));
    }

    template <typename E>
    static constexpr E expected_op(E lhs, size_t rhs) noexcept
    requires (rotop == dpp::rotl)
    {
        return static_cast<E>(dpl::rotl(dpl::to_unsigned(lhs), rhs));
    }

    template <typename E>
    static constexpr size_t test_count() noexcept {
        auto lanes = abi_traits<E>::size();
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
        return max < limit ? max : limit;
    }

    template <typename E>
    static consteval auto make_const_bits() noexcept {
        constexpr auto const_count = test_count<E>();
        return dpl::apply(
            [const_count](auto... idx) {
                constexpr auto lanes = dpp::simd_abi_traits<A, E>::size();
                using bitset_t = dpl::bitset<lanes>;
                using array_t = array<bitset_t, const_count>;

                test::mt19937 rng; // TODO consteval RNG?
                dpl::test::bit_generator<lanes> bitgen;
                return array_t{(dpl::ignore = idx, bitgen(rng))...};
            },
            dpl::make_index_sequence<const_count>{});
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return bitrot::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        dpl::test::array_generator<A, E> const data_generator;
        using shift_t = dpl::make_unsigned_t<E>;
        dpl::test::array_generator<A, shift_t> const shift_generator;
        mask_generator<A, E> mask_generator;
        auto const lhs = data_generator(engine);
        auto const rhs = shift_generator(engine);
        auto const src = data_generator(engine);
        auto const vmask = mask_generator(engine);

        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op(lhs[i], rhs[i]);
        }

        test::operation_fixture<A>::test(
            test::bitcmp, expected, rotop, lhs, rhs);
        test::operation_fixture<A>::test_masked(rotop, src, vmask, lhs, rhs);
        test::operation_fixture<A>::test_masked(
            rotop, dpp::zero, vmask, lhs, rhs);

        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto count = test_count<E>();
            constexpr auto masks = make_const_bits<E>();
            dpl::pack::for_each(
                [&]<size_t I>(dpl::size_constant<I>) {
                    auto const lhs = data_generator(engine);
                    auto const rhs = shift_generator(engine);
                    auto const src = data_generator(engine);
                    constexpr auto mask = masks[I];
                    using cmask_t =
                        dpp::const_mask<mask.size(), dpl::to_underlying(mask)>;
                    constexpr cmask_t cmask;

                    test::operation_fixture<A>::test_masked(
                        rotop, src, cmask, lhs, rhs);
                    test::operation_fixture<A>::test_masked(
                        rotop, dpp::zero, cmask, lhs, rhs);
                },
                dpl::make_index_sequence<count>{});
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto count = test_count<E>();
            dpl::pack::for_each(
                [&]<size_t I, size_t S = dpl::type_bit_v<E> / count>(
                    dpl::size_constant<I>, dpl::size_constant<S> = {}) {
                    constexpr auto offset = []() {
                        test::mt19937 rng{};
                        return test::scalar_generator<size_t>(0zu, S)(rng);
                    }();

                    constexpr auto crhs = dpp::imm<I * S + offset>;

                    for (auto i = 0zu; i < expected.size(); ++i) {
                        expected[i] = expected_op(lhs[i], crhs);
                    }

                    test::operation_fixture<A>::test(
                        test::bitcmp, expected, rotop, lhs, crhs);
                    test::operation_fixture<A>::test_masked(
                        rotop, src, vmask, lhs, crhs);
                    test::operation_fixture<A>::test_masked(
                        rotop, dpp::zero, vmask, lhs, crhs);
                },
                dpl::make_index_sequence<count>{});
        }

        for (auto i = 0zu; i < test_count<E>(); ++i) {
            auto const srhs = shift_generator.scalar(engine);
            for (auto j = 0zu; j < expected.size(); ++j) {
                expected[j] = expected_op(lhs[j], srhs);
            }

            test::operation_fixture<A>::test(
                test::bitcmp, expected, rotop, lhs, srhs);
            test::operation_fixture<A>::test_masked(
                rotop, src, vmask, lhs, srhs);
            test::operation_fixture<A>::test_masked(
                rotop, dpp::zero, vmask, lhs, srhs);
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using rotl = bitrot<dpp::rotl, A>;
export template <dpp::simd_abi A>
using rotr = bitrot<dpp::rotr, A>;

} // namespace dpl::test
