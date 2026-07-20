// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>
#include <cstdlib>

export module dpl.test:bitwise;
export import :binary_transform;
export import :unary_transform;
export import :utils.span;
export import :utils.comparison;
export import :utils.bitset_helpers;
export import :data_generator;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class bitwise {
    using abi_t = A;
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;

    template <auto Op>
    struct expected_op_t {
        template <typename E>
        static constexpr E operator()(E lhs, E rhs) noexcept
        requires (Op != dpp::bwshift_left && Op != dpp::bwshift_right &&
            Op != dpp::bwnot)
        {
            using bit_type = dpl::bitset<dpl::type_bit_v<E>>;
            if constexpr (Op == dpp::bwand) {
                return dpl::bit_cast<E>(dpl::bit_cast<bit_type>(lhs) &
                    dpl::bit_cast<bit_type>(rhs));
            } else if constexpr (Op == dpp::bwor) {
                return dpl::bit_cast<E>(dpl::bit_cast<bit_type>(lhs) |
                    dpl::bit_cast<bit_type>(rhs));
            } else if constexpr (Op == dpp::bwandnot) {
                return dpl::bit_cast<E>(dpl::bit_cast<bit_type>(lhs) &
                    ~dpl::bit_cast<bit_type>(rhs));
            } else if constexpr (Op == dpp::bwornot) {
                return dpl::bit_cast<E>(dpl::bit_cast<bit_type>(lhs) |
                    ~dpl::bit_cast<bit_type>(rhs));
            } else {
                static_assert(Op == dpp::bwxor);
                return dpl::bit_cast<E>(dpl::bit_cast<bit_type>(lhs) ^
                    dpl::bit_cast<bit_type>(rhs));
            }
        }

        template <typename E>
        static constexpr E operator()(E arg) noexcept
        requires (Op == dpp::bwnot)
        {
            using bit_type = dpl::bitset<dpl::type_bit_v<E>>;
            return dpl::bit_cast<E>(~dpl::bit_cast<bit_type>(arg));
        }

        template <typename E>
        static constexpr E operator()(E lhs, size_t rhs) noexcept
        requires (Op == dpp::bwshift_left)
        {
            using bit_type = dpl::bitset<dpl::type_bit_v<E>>;
            return dpl::bit_cast<E>(dpl::bit_cast<bit_type>(lhs) << rhs);
        }

        template <typename E>
        static constexpr E operator()(E lhs, size_t rhs) noexcept
        requires (Op == dpp::bwshift_right)
        {
            if constexpr (dpl::integral<E>) {
                return static_cast<E>(lhs >> rhs);
            } else {
                using bit_type = dpl::bitset<dpl::type_bit_v<E>>;
                return dpl::bit_cast<E>(dpl::bit_cast<bit_type>(lhs) >> rhs);
            }
        }
    };

    template <auto bwop>
    static constexpr expected_op_t<bwop> expected_op{};

    template <typename E>
    static constexpr E max = []() {
        if constexpr (dpl::integral<E>) {
            constexpr auto shift = dpl::type_bit_v<E> / 2;
            return dpl::integral_traits<E>::max_value >> shift;
        } else {
            constexpr auto bias = dpl::floating_point_traits<E>::exponent_bias;
            constexpr auto exp = dpl::floating_point_traits<E>::exponent_mask;
            constexpr auto shift = dpl::countr_zero(exp);
            // minus 1 for reduced range, so that sqrt * sqrt < max_value
            constexpr auto sqrt =
                dpl::bitset<dpl::type_bit_v<E>>(
                    ((dpl::to_underlying(exp >> shift) - bias) >> 1) + bias - 1)
                << shift;
            return dpl::bit_cast<E>(sqrt);
        }
    }();

    template <typename E>
    static constexpr E min = []() {
        if constexpr (dpl::integral<E>) {
            constexpr auto shift = dpl::type_bit_v<E> / 2;
            return dpl::integral_traits<E>::min_value >> shift;
        } else {
            return -max<E>;
        }
    }();

public:
    template <dpp::simd_primitive_operation auto bwop, rng_like Rng,
        dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return bitwise::template run<E, bwop>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E,
        dpp::simd_primitive_operation auto bwop, rng_like Rng>
    requires (bwop != dpp::bwshift_left && bwop != dpp::bwshift_right &&
        bwop != dpp::bwnot)
    static constexpr bool run(Rng& engine) {
        dpl::test::array_generator<abi_t, E> const data_generator(
            min<E>, max<E>);
        auto const lhs = data_generator(engine);
        auto const rhs = data_generator(engine);
        auto const src = max<E> * max<E>;

        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op<bwop>(lhs[i], rhs[i]);
        }

        dpl::test::binary_transform<abi_t>::template test<E>(
            lhs, rhs, bwop, expected, test::bitcmp);
        dpl::test::binary_transform<abi_t>::template test_masked<E>(
            lhs, rhs, bwop, src);

        return true;
    }

    template <dpp::simd_element_for<A> E,
        dpp::simd_primitive_operation auto bwop, rng_like Rng>
    requires (bwop == dpp::bwshift_left || bwop == dpp::bwshift_right)
    static constexpr bool run(Rng& engine) {
        using shift_t = dpp::signed_representation_t<E>;
        constexpr E offset =
            bwop == dpp::bwshift_left ? max<E> * max<E> : static_cast<E>(0);
        dpl::test::array_generator<abi_t, E> const data_generator(
            min<E> + offset, max<E> + offset);
        dpl::test::array_generator<abi_t, shift_t> const shift_generator(
            0, dpl::type_bit_v<E>);

        constexpr E src_min = bwop == dpp::bwshift_left ? min<E> : max<E>;
        constexpr E src_max =
            bwop == dpp::bwshift_left ? min<E> + offset : dpp::max_value_v<E>;
        dpl::test::scalar_generator<E> const src_generator(src_min, src_max);
        auto const lhs = data_generator(engine);
        auto const rhs = shift_generator(engine);
        auto const src = src_generator(engine);

        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op<bwop>(lhs[i], rhs[i]);
        }

        dpl::test::binary_transform<abi_t>::template test<E>(
            lhs, rhs, bwop, expected, test::bitcmp);
        dpl::test::binary_transform<abi_t>::template test_masked<E>(
            lhs, rhs, bwop, src);

        dpl::pack::for_each(
            [&](auto idx) {
                auto const shifteri = [idx](auto... args) {
                    return bwop(args..., idx);
                };
                auto const shifter = [idx](auto... args) {
                    return bwop(args..., idx());
                };

                for (auto i = 0zu; i < expected.size(); ++i) {
                    expected[i] = expected_op<bwop>(lhs[i], idx);
                }

                dpl::test::unary_transform<abi_t>::template test<E>(
                    lhs, shifteri, expected, test::bitcmp);
                dpl::test::unary_transform<abi_t>::template test_masked<E>(
                    lhs, shifteri, src);
                dpl::test::unary_transform<abi_t>::template test<E>(
                    lhs, shifter, expected, test::bitcmp);
                dpl::test::unary_transform<abi_t>::template test_masked<E>(
                    lhs, shifter, src);
            },
            dpl::make_index_sequence<dpl::type_bit_v<E>>{});

        return true;
    }

    template <dpp::simd_element_for<A> E,
        dpp::simd_primitive_operation auto bwop, rng_like Rng>
    requires (bwop == dpp::bwnot)
    static constexpr bool run(Rng& engine) {
        dpl::test::array_generator<abi_t, E> const data_generator(
            dpl::test::half_range);
        dpl::test::scalar_generator<E> const src_generator(
            dpp::max_value_v<E> / 4 * 3, dpp::max_value_v<E>);

        auto const data = data_generator(engine);
        auto const src = src_generator(engine);
        auto expected = data;
        for (auto& val : expected) {
            val = expected_op<bwop>(val);
        }

        dpl::test::unary_transform<abi_t>::template test<E>(
            data, bwop, expected, test::bitcmp);
        dpl::test::unary_transform<abi_t>::template test_masked<E>(
            data, bwop, src);
    }
};
} // namespace dpl::test
