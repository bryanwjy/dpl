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
        auto const lhs = data_generator(engine);
        auto const rhs = shift_generator(engine);
        auto const src =
            -lhs[dpl::test::scalar_generator<shift_t>{}(engine) % lhs.size()];

        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op(lhs[i], rhs[i]);
        }

        dpl::test::unary_transform<A>::template test<E>(
            lhs,
            [rhs = dpp::load<shift_t, A>(rhs.data())](
                auto... args) { return rotop(args..., rhs); },
            expected, test::bitcmp);
        dpl::test::unary_transform<A>::template test_masked<E>(
            lhs,
            [rhs = dpp::load<shift_t, A>(rhs.data())](
                auto... args) { return rotop(args..., rhs); },
            src);

        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto count = 8zu;
            dpl::pack::for_each(
                [&]<size_t I, size_t S = dpl::type_bit_v<E> / count>(
                    dpl::size_constant<I>, dpl::size_constant<S> = {}) {
                    constexpr auto offset = []() {
                        test::mt19937 rng{};
                        return test::scalar_generator<size_t>(0zu, S)(rng);
                    }();
                    auto const rotatei = [offset](auto... args) {
                        return rotop(args..., dpp::imm<I * S + offset>);
                    };
                    for (auto i = 0zu; i < expected.size(); ++i) {
                        expected[i] = expected_op(lhs[i], I * S + offset);
                    }

                    dpl::test::unary_transform<A>::template test<E>(
                        lhs, rotatei, expected, test::bitcmp);
                    dpl::test::unary_transform<A>::template test_masked<E>(
                        lhs, rotatei, src);
                },
                dpl::make_index_sequence<count>{});
        }

        for (auto i = 0zu; i < dpl::type_bit_v<E>; ++i) {
            if consteval {
                // reduce compile time
                auto const rand = shift_generator.scalar(engine);
                if (rand % 7 > 4) {
                    break;
                }
            }

            for (auto j = 0zu; j < expected.size(); ++j) {
                expected[j] = expected_op(lhs[j], i);
            }

            auto const rotate = [i](auto... args) { return rotop(args..., i); };

            dpl::test::unary_transform<A>::template test<E>(
                lhs, rotate, expected, test::bitcmp);
            dpl::test::unary_transform<A>::template test_masked<E>(
                lhs, rotate, src);
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using rotl = bitrot<dpp::rotl, A>;
export template <dpp::simd_abi A>
using rotr = bitrot<dpp::rotr, A>;

} // namespace dpl::test
