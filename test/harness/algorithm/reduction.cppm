// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.reduction;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_algorithm_operation auto rop, dpp::simd_abi A,
    dpp::simd_element_for<A> E>
struct reduction_data_generator : test::array_generator<A, E> {
    using test::array_generator<A, E>::array_generator;
    using test::array_generator<A, E>::operator();
};

export template <dpp::simd_algorithm_operation auto rop, dpp::simd_abi A,
    dpp::simd_element_for<A> E>
requires (rop == dpp::hsum && dpl::integral<E>)
struct reduction_data_generator<rop, A, E> {
    template <typename Rng>
    constexpr auto operator()(Rng& engine) const noexcept {
        return gen(engine);
    }

private:
    static constexpr E range =
        dpl::integral_traits<E>::max_value / dpp::simd_abi_traits<A, E>::size;
    test::array_generator<A, E> gen =
        test::array_generator<A, E>(dpl::is_signed_v<E> ? -range : 0, range);
};

export template <dpp::simd_algorithm_operation auto rop, dpp::simd_abi A,
    dpp::simd_element_for<A> E>
requires (rop == dpp::hsum && dpl::floating_point_like<E>)
struct reduction_data_generator<rop, A, E> {

public:
    template <typename Rng>
    constexpr auto operator()(Rng& engine) const noexcept {
        auto const is_neg = sign_gen(engine);
        auto const frac = frac_gen(engine);
        using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
        auto const exp = dpl::bit_cast<uint_t>(
            static_cast<bitset_t>(exp_gen(engine))
            << dpl::countr_zero(dpl::floating_point_traits<E>::exponent_mask));

        constexpr auto shift = dpl::type_bit_v<E> - 1;
        auto result = result_gen(engine);
        for (auto i = 0zu; i < frac.size(); ++i) {
            auto const val =
                static_cast<uint_t>((is_neg << shift) | frac[i] | exp);
            result[i] = __DPL bit_cast<E>(val);
        }

        return result;
    }

private:
    using uint_t = dpp::unsigned_representation_t<E>;
    static constexpr auto nexp =
        dpl::popcount(dpl::floating_point_traits<E>::exponent_mask);
    static constexpr uint_t frac_range =
        dpl::to_underlying(dpl::floating_point_traits<E>::mantissa_mask) /
        dpp::simd_abi_traits<A, E>::size;

    test::bit_generator<nexp> exp_gen;
    test::scalar_generator<uint_t> sign_gen =
        test::scalar_generator<uint_t>(0, 2);
    test::array_generator<A, uint_t> frac_gen =
        test::array_generator<A, uint_t>(0, frac_range);
    test::array_generator<A, E> result_gen;
};

export template <dpp::simd_abi A, dpp::simd_algorithm_operation auto rop>
class reduction {
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using data = test::array<E, abi_traits<E>::size>;

    template <dpp::simd_element_for<A> E>
    static constexpr E expected_op(data<E> const& val) noexcept {
        auto result = [&] {
            if constexpr (dpl::floating_point_like<E> && rop == dpp::hsum) {
                // TODO deal with MSVC
                return static_cast<long double>(val[0]);
            } else {
                return val[0];
            }
        }();
        for (auto i = 1zu; i < val.size(); ++i) {
            if constexpr (rop == dpp::hsum) {
                result += val[i];
            } else if constexpr (rop == dpp::hmax) {
                result = result < val[i] ? val[i] : result;
            } else {
                result = result > val[i] ? val[i] : result;
            }
        }

        return static_cast<E>(result);
    }

    template <dpp::simd_element_for<A> E>
    static constexpr E expected_op(
        data<E> const& val, dpl::bitset<abi_traits<E>::size> mask) noexcept
    requires dpp::fixed_width_abi<A>
    {
        auto const first = dpl::countr_zero(mask);
        auto result = [&] {
            if constexpr (dpl::floating_point_like<E> && rop == dpp::hsum) {
                // TODO deal with MSVC
                return static_cast<long double>(val[first]);
            } else {
                return val[first];
            }
        }();
        for (auto i = first + 1; i < val.size(); ++i) {
            if (mask[i]) {
                if constexpr (rop == dpp::hsum) {
                    result += val[i];
                } else if constexpr (rop == dpp::hmax) {
                    result = result < val[i] ? val[i] : result;
                } else {
                    result = result > val[i] ? val[i] : result;
                }
            }
        }

        return result;
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return reduction::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires dpp::fixed_width_abi<A>
    {
        constexpr auto width = abi_traits<E>::size();
        reduction_data_generator<rop, A, E> const data_generator;
        test::bit_generator<width> mask_generator;
        auto const loop_count = []() {
            if consteval {
                return 3;
            } else {
                return 128;
            }
        }();

        for (auto i = 0; i < loop_count; ++i) {
            auto const val = data_generator(engine);
            auto const mask = [&] {
                auto mask = mask_generator(engine);
                while (!mask) {
                    mask = mask_generator(engine);
                }
                return mask;
            }();
            auto const vval = dpp::load<A>(val.data());
            auto const vmask = dpp::from_bitset<A, E>(mask);

            {
                auto const expected = expected_op(val);
                auto const actual = rop(vval);
                if constexpr (dpl::is_integral_v<E>) {
                    assert(test::bitcmp(actual, expected));
                } else {
                    using sint_t = dpp::signed_representation_t<E>;
                    auto diff = __DPL bit_cast<sint_t>(actual) -
                        __DPL bit_cast<sint_t>(expected);
                    diff = diff < 0 ? -diff : diff;
                    assert(diff < 2); // less than 2 ulp
                }
            }
            {
                auto const expected = expected_op(val, mask);
                auto const actual = rop(vval, vmask);
                if constexpr (dpl::is_integral_v<E>) {
                    assert(test::bitcmp(actual, expected));
                } else {
                    using sint_t = dpp::signed_representation_t<E>;
                    auto diff = __DPL bit_cast<sint_t>(actual) -
                        __DPL bit_cast<sint_t>(expected);
                    diff = diff < 0 ? -diff : diff;
                    assert(diff < 2); // less than 2 ulp
                }
            }
        }

        {
            constexpr auto count = 3zu;
            constexpr auto masks = []() {
                return dpl::apply(
                    [](auto... idx) {
                        test::mt19937 rng{};
                        using bitset_t = dpl::bitset<abi_traits<E>::size()>;
                        return array<dpl::bitset<abi_traits<E>::size()>, count>{
                            (dpl::test::bit_generator<abi_traits<E>::size() +
                                 idx * 0>()(rng) |
                                bitset_t(dpl::low_bits, 1))...};
                    },
                    dpl::make_index_sequence<count>{});
            }();

            dpl::pack::for_each(
                [&]<size_t I>(dpl::size_constant<I>) {
                    auto const val = data_generator(engine);
                    auto const cmask = dpp::const_mask<masks[I].size(),
                        dpl::to_underlying(masks[I])>{};

                    auto const vval = dpp::load<A>(val.data());
                    {
                        auto const expected = expected_op(val, masks[I]);
                        auto const actual = rop(vval, cmask);
                        if constexpr (dpl::is_integral_v<E>) {
                            assert(test::bitcmp(actual, expected));
                        } else {
                            using sint_t = dpp::signed_representation_t<E>;
                            auto diff = __DPL bit_cast<sint_t>(actual) -
                                __DPL bit_cast<sint_t>(expected);
                            diff = diff < 0 ? -diff : diff;
                            assert(diff < 2); // less than 2 ulp
                        }
                    }
                },
                dpl::make_index_sequence<count>{});
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using hsum = reduction<A, dpp::hsum>;
export template <dpp::simd_abi A>
using hmin = reduction<A, dpp::hmin>;
export template <dpp::simd_abi A>
using hmax = reduction<A, dpp::hmax>;

export template <dpp::simd_abi A>
class reduce {

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return reduce::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        auto const loop_count = []() {
            if consteval {
                return 3;
            } else {
                return 128;
            }
        }();

        reduction_data_generator<dpp::hmax, A, E> const max_generator;
        reduction_data_generator<dpp::hmin, A, E> const min_generator;

        for (auto i = 0; i < loop_count; ++i) {

            // Summation introduces error complexity, ignore it for now
            // This is anyway just a check for the generic reduction
            {
                auto const val = max_generator(engine);
                auto const vval = dpp::load<A>(val.data());
                auto const expected = dpp::hmax(vval);
                auto const actual = dpp::reduce(vval, dpp::max);
                assert(test::bitcmp(actual, expected));
            }
            {
                auto const val = min_generator(engine);
                auto const vval = dpp::load<A>(val.data());
                auto const expected = dpp::hmin(vval);
                auto const actual = dpp::reduce(vval, dpp::min);
                assert(test::bitcmp(actual, expected));
            }
        }

        return true;
    }
};

} // namespace dpl::test
