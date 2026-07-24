// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>
#include <cstdlib>

export module dpl.test.harness.operations.bitwise_mask;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class bitwise_mask {
    using abi_t = A;
    template <typename E>
    using mask_t = dpp::basic_mask<E, A>;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<abi_t, E>;
    template <typename E>
    // Only works for fixed_width
    using bitset_t = dpl::bitset<abi_traits<E>::size>;

    template <auto Op>
    struct expected_op_t {
        template <size_t N>
        static constexpr dpl::bitset<N> operator()(
            dpl::bitset<N> lhs, dpl::bitset<N> rhs) noexcept
        requires (Op != dpp::bwshift_left && Op != dpp::bwshift_right &&
            Op != dpp::bwnot)
        {
            if constexpr (Op == dpp::bwand) {
                return lhs & rhs;
            } else if constexpr (Op == dpp::bwor) {
                return lhs | rhs;
            } else if constexpr (Op == dpp::bwandnot) {
                return lhs & ~rhs;
            } else if constexpr (Op == dpp::bwornot) {
                return lhs | ~rhs;
            } else {
                static_assert(Op == dpp::bwxor);
                return lhs ^ rhs;
            }
        }

        template <size_t N>
        static constexpr dpl::bitset<N> operator()(dpl::bitset<N> arg) noexcept
        requires (Op == dpp::bwnot)
        {
            return ~arg;
        }

        template <size_t N>
        static constexpr dpl::bitset<N> operator()(
            dpl::bitset<N> lhs, size_t rhs) noexcept
        requires (Op == dpp::bwshift_left || Op == dpp::bwshift_right)
        {
            if constexpr (Op == dpp::bwshift_right) {
                return lhs >> rhs;
            } else {
                return lhs << rhs;
            }
        }
    };

    template <auto bwop>
    static constexpr expected_op_t<bwop> expected_op;

    template <typename E>
    static constexpr auto equal(mask_t<E> actual, bitset_t<E> expected) noexcept
    requires dpp::fixed_width_abi<abi_t>
    {
        return dpp::to_bitset(actual) == expected &&
            dpp::all_of(actual == dpp::from_bitset<abi_t, E>(expected)) &&
            actual.size() == expected.size();
    }

public:
    template <dpp::simd_primitive_operation auto bwop, rng_like Rng,
        dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return bitwise_mask::template run<E, bwop>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E,
        dpp::simd_primitive_operation auto bwop, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires dpp::fixed_width_abi<abi_t>
    {
        auto const data_generator = []() {
            if constexpr (dpl::integral_bitset_type<bitset_t<E>>) {
                using type = typename bitset_t<E>::underlying_type;
                return [gen = dpl::test::scalar_generator<type>{}](
                           Rng& engine) { return bitset_t<E>(gen(engine)); };
            } else {
                return
                    [gen = dpl::test::scalar_generator<size_t>{}](Rng& engine) {
                        using chunk_t = dpl::bitset<dpl::type_bit_v<size_t>>;
                        constexpr auto chunks =
                            bitset_t<E>::size() / dpl::type_bit_v<size_t>;
                        return dpl::apply(
                            [&](auto... idx) {
                                chunk_t const args[sizeof...(idx)] = {
                                    ((void)idx, chunk_t(gen(engine)))...};
                                return bitset_t<E>(args[idx]...);
                            },
                            dpl::make_index_sequence<chunks>{});
                    };
            }
        }();
        if constexpr (bwop == dpp::bwshift_left || bwop == dpp::bwshift_right) {
            auto const lhs = data_generator(engine);
            dpl::test::scalar_generator<size_t> const shift_generator(
                0zu, mask_t<E>::size());

            for (auto i = 0zu; i < abi_traits<E>::size(); ++i) {
                if consteval {
                    if (abi_traits<E>::size() > 16 && i % 3 > 0) {
                        break;
                    }
                }

                auto const expected = expected_op<bwop>(lhs, i);
                assert(
                    equal(bwop(dpp::from_bitset<abi_t, E>(lhs), i), expected));
            }

            // Shifting out of range produces zero
            assert(equal(
                bwop(dpp::from_bitset<abi_t, E>(lhs), abi_traits<E>::size()),
                bitset_t<E>()));
            assert(equal(bwop(dpp::from_bitset<abi_t, E>(lhs),
                             abi_traits<E>::size() + 1),
                bitset_t<E>()));
            assert(equal(bwop(dpp::from_bitset<abi_t, E>(lhs),
                             abi_traits<E>::size() + 2),
                bitset_t<E>()));
            assert(dpl::pack::all_of(
                [&](auto idx) {
                    return equal(bwop(dpp::from_bitset<abi_t, E>(lhs),
                                     dpp::imm<idx() + abi_traits<E>::size()>),
                        bitset_t<E>());
                },
                dpl::make_index_sequence<4>{}));

            return dpl::pack::all_of(
                [&](auto idx) {
                    if consteval {
                        if constexpr (abi_traits<E>::size() > 16 &&
                            idx() % 3 > 0) {
                            return true;
                        }
                    }

                    auto const expected = expected_op<bwop>(lhs, idx);
                    return equal(
                        bwop(dpp::from_bitset<abi_t, E>(lhs), idx), expected);
                },
                dpl::make_index_sequence<abi_traits<E>::size()>{});
        } else if constexpr (bwop == dpp::bwnot) {
            auto const arg = data_generator(engine);
            using result_type = decltype(bwop(dpp::from_bitset<abi_t, E>(arg)));
            static_assert(dpl::same_as<result_type, mask_t<E>>);
            auto const expected = expected_op<bwop>(arg);
            return equal(bwop(dpp::from_bitset<abi_t, E>(arg)), expected);
        } else {
            if constexpr (bwop != dpp::bwshift_left &&
                bwop != dpp::bwshift_right && bwop != dpp::bwnot) {}
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const expected = expected_op<bwop>(lhs, rhs);
            return equal(bwop(dpp::from_bitset<abi_t, E>(lhs),
                             dpp::from_bitset<abi_t, E>(rhs)),
                expected);
        }
    }
};
} // namespace dpl::test
