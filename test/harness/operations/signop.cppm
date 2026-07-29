// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.signop;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class signop {
    using abi_t = A;
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;
    template <auto op>
    struct expected_op_t {
        template <typename E>
        static constexpr E operator()(E arg) noexcept
        requires (op == dpp::abs)
        {
            if constexpr (dpl::unsigned_integral<E>) {
                return arg;
            } else if constexpr (dpl::integral<E>) {
                using U = dpp::unsigned_representation_t<E>;
                return arg < E(0) ? static_cast<E>(-static_cast<U>(arg)) : arg;
            } else {
                constexpr auto signbit = dpl::floating_point_traits<E>::signbit;
                return dpl::bit_cast<E>(
                    dpl::to_bit_representation(arg) & ~signbit);
            }
        }

        template <typename E>
        static constexpr E operator()(E arg) noexcept
        requires (op == dpp::negate)
        {
            if constexpr (dpl::integral<E>) {
                return -arg;
            } else {
                constexpr auto signbit = dpl::floating_point_traits<E>::signbit;
                return dpl::bit_cast<E>(
                    dpl::to_bit_representation(arg) ^ signbit);
            }
        }
    };

    template <auto op>
    static constexpr expected_op_t<op> expected_op{};

public:
    template <dpp::simd_primitive_operation auto op, rng_like Rng,
        dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return signop::template run<E, op>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, dpp::simd_primitive_operation auto op,
        rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<abi_t, E> const data_generator(test::half_range);
        test::scalar_generator<E> const src_generator(
            dpp::max_value_v<E> / 4 * 3, dpp::max_value_v<E>);

        auto const data = data_generator(engine);
        auto const src = src_generator(engine);
        auto expected = data;
        for (auto& val : expected) {
            val = expected_op<op>(val);
        }

        test::unary_transform<abi_t>::template test<E>(data, op, expected);
        test::unary_transform<abi_t>::template test_masked<E>(data, op, src);

        // INT_MIN: abs(INT_MIN) == INT_MIN on 2's complement hardware (wraps).
        // Float special cases (-0.0, ±inf, NaN) compared via bit_cast since
        // NaN != NaN under IEEE 754.
        if constexpr (dpl::signed_integral<E>) {
            if not consteval {
                // Implementation-defined
                auto const min = dpl::integral_traits<E>::min_value;
                test::unary_transform<abi_t>::test(min, op, min);
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            for (auto const arg :
                test::array{dpp::msb_v<E>, -dpp::infinity_v<E>,
                    dpp::infinity_v<E>, -dpp::nan_v<E>, dpp::nan_v<E>}) {

                auto const expected = expected_op<op>(arg);
                test::unary_transform<abi_t>::test(
                    arg, op, expected, test::bitcmp);
            }
        }

        return true;
    }
};

} // namespace dpl::test
