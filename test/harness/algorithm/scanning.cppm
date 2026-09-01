// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.scanning;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_algorithm_operation auto scanop, dpp::simd_abi A,
    dpp::simd_element_for<A> E>
struct scanning_data_generator : test::array_generator<A, E> {
    using test::array_generator<A, E>::array_generator;
    using test::array_generator<A, E>::operator();
    using test::array_generator<A, E>::scalar;
};

export template <dpp::simd_algorithm_operation auto scanop, dpp::simd_abi A,
    dpp::simd_element_for<A> E>
requires (scanop == dpp::exscan_sum || scanop == dpp::scan_sum) &&
    dpl::integral<E>
struct scanning_data_generator<scanop, A, E> {
    template <typename Rng>
    constexpr auto operator()(Rng& engine) const noexcept {
        return gen(engine);
    }

    template <typename Rng>
    constexpr E scalar(Rng& engine) const noexcept(dpp::fixed_width_abi<A>) {
        return gen.scalar(engine);
    }

private:
    static constexpr E range =
        dpl::integral_traits<E>::max_value / dpp::simd_abi_traits<A, E>::size;
    test::array_generator<A, E> gen =
        test::array_generator<A, E>(dpl::is_signed_v<E> ? -range : 0, range);
};

export template <dpp::simd_algorithm_operation auto scanop, dpp::simd_abi A,
    dpp::simd_element_for<A> E>
requires (scanop == dpp::exscan_sum || scanop == dpp::scan_sum) &&
    dpl::floating_point_like<E>
struct scanning_data_generator<scanop, A, E> {

public:
    template <typename Rng>
    constexpr auto operator()(Rng& engine) const
        noexcept(dpp::fixed_width_abi<A>) {
        auto const is_neg = sign_gen(engine);
        auto const frac = frac_gen(engine);
        using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
        auto const exp = dpl::bit_cast<uint_t>(
            static_cast<bitset_t>(exp_gen(engine))
            << dpl::countr_zero(dpl::floating_point_traits<E>::exponent_mask));

        constexpr auto shift = dpl::type_bit_v<E> - 1;
        auto result = make_result();
        for (auto i = 0zu; i < frac.size(); ++i) {
            auto const val =
                static_cast<uint_t>((is_neg << shift) | frac[i] | exp);
            result[i] = __DPL bit_cast<E>(val);
        }

        return result;
    }

    template <typename Rng>
    constexpr E scalar(Rng& engine) const noexcept(dpp::fixed_width_abi<A>) {
        auto const is_neg = sign_gen(engine);
        auto const frac = frac_gen.scalar(engine);
        using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
        auto const exp = dpl::bit_cast<uint_t>(
            static_cast<bitset_t>(exp_gen(engine))
            << dpl::countr_zero(dpl::floating_point_traits<E>::exponent_mask));

        constexpr auto shift = dpl::type_bit_v<E> - 1;
        auto const val = static_cast<uint_t>((is_neg << shift) | frac | exp);
        return __DPL bit_cast<E>(val);
    }

private:
    static constexpr auto make_result() noexcept(dpp::fixed_width_abi<A>) {
        if constexpr (dpp::fixed_width_abi<A>) {
            return array<E, dpp::simd_abi_traits<A, E>::size>{};
        } else {
            return dynamic_array<E>(dpp::simd_abi_traits<A, E>::size());
        }
    }

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
};

template <auto op, typename E>
consteval E scan_identity() noexcept {
    if constexpr (op == dpp::exscan_sum || op == dpp::scan_sum) {
        return 0;
    } else if constexpr (op == dpp::exscan_min || op == dpp::scan_min) {
        if constexpr (dpl::floating_point_like<E>) {
            return dpp::infinity_v<E>;
        } else {
            return dpp::max_value_v<E>;
        }
    } else if constexpr (dpl::is_integral_v<E>) {
        return dpp::min_value_v<E>;
    } else {
        return -dpp::infinity_v<E>;
    }
}

export template <dpp::simd_abi A, dpp::simd_algorithm_operation auto scanop>
class exclusive_scan {

    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <dpp::simd_element_for<A> E, typename T>
    static constexpr T expected_op(T const& vals, E init) noexcept {
        constexpr auto float_sum =
            dpl::floating_point_like<E> && scanop == dpp::exscan_sum;
        auto result = [] {
            using R = dpl::conditional_t<float_sum, long double, E>;
            if constexpr (dpp::fixed_width_abi<A>) {
                return array<R, abi_traits<E>::size>{};
            } else {
                return dynamic_array<R>(abi_traits<E>::size());
            }
        }();
        for (auto& val : result) {
            val = init;
        }

        for (auto i = 1zu; i < vals.size(); ++i) {
            auto const j = i - 1;
            if constexpr (scanop == dpp::exscan_sum) {
                result[i] = result[j] + vals[j];
            } else if constexpr (scanop == dpp::exscan_max) {
                result[i] = result[j] < vals[j] ? vals[j] : result[j];
            } else {
                result[i] = result[j] > vals[j] ? vals[j] : result[j];
            }
        }

        if constexpr (!float_sum) {
            return result;
        } else if constexpr (dpp::fixed_width_abi<A>) {
            auto output = array<E, abi_traits<E>::size>{};
            output[0] = init;
            for (auto i = 1zu; i < vals.size(); ++i) {
                output[i] = result[i];
            }
            return output;
        } else {
            auto output = dynamic_array<E>(abi_traits<E>::size());
            output[0] = init;
            for (auto i = 1zu; i < vals.size(); ++i) {
                output[i] = result[i];
            }
            return output;
        }
    }

    template <dpp::simd_element_for<A> E, typename T, typename M>
    static constexpr T expected_op(T const& vals, M mask, E init) noexcept {
        auto clean(vals);
        for (auto i = 0zu; i < vals.size(); ++i) {
            if (!mask[i]) {
                clean[i] = scan_identity<scanop, E>();
            }
        }
        return expected_op(clean, init);
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return exclusive_scan::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires dpp::fixed_width_abi<A>
    {
        constexpr auto width = abi_traits<E>::size();
        scanning_data_generator<scanop, A, E> const data_generator;
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
            auto const init = data_generator.scalar(engine);
            auto const mask = mask_generator(engine);
            auto const vval = dpp::load<A>(val.data());
            auto const vmask = dpp::from_bitset<A, E>(mask);

            {
                auto const expected = expected_op(val, init);
                auto const vexpected = dpp::load<A, E>(expected.data());
                auto const vactual = scanop(vval, init);
                if constexpr (dpl::is_integral_v<E>) {
                    assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                } else {
                    using sint_t = dpp::signed_representation_t<E>;
                    auto const diff = dpp::reinterpret<sint_t>(vactual) -
                        dpp::reinterpret<sint_t>(vexpected);
                    // less than 2 ulp
                    assert(dpp::all_of(dpp::cmplt(dpp::abs(diff), 2)));
                }
            }
            {
                auto const expected = expected_op(val, mask, init);
                auto const vexpected = dpp::load<A, E>(expected.data());
                auto const vactual = scanop(vval, vmask, init);
                if constexpr (dpl::is_integral_v<E>) {
                    assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                } else {
                    using sint_t = dpp::signed_representation_t<E>;
                    auto const diff = dpp::reinterpret<sint_t>(vactual) -
                        dpp::reinterpret<sint_t>(vexpected);
                    // less than 2 ulp
                    assert(dpp::all_of(dpp::cmplt(dpp::abs(diff), 2)));
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
                    auto const init = data_generator.scalar(engine);
                    auto const cmask = dpp::const_mask<masks[I].size(),
                        dpl::to_underlying(masks[I])>{};

                    auto const vval = dpp::load<A>(val.data());
                    {
                        auto const expected = expected_op(val, masks[I], init);
                        auto const vexpected = dpp::load<A, E>(expected.data());
                        auto const vactual = scanop(vval, cmask, init);
                        if constexpr (dpl::is_integral_v<E>) {
                            assert(
                                dpp::all_of(test::bitcmp(vactual, vexpected)));
                        } else {
                            using sint_t = dpp::signed_representation_t<E>;
                            auto const diff =
                                dpp::reinterpret<sint_t>(vactual) -
                                dpp::reinterpret<sint_t>(vexpected);
                            // less than 2 ulp
                            assert(dpp::all_of(dpp::cmplt(dpp::abs(diff), 2)));
                        }
                    }
                },
                dpl::make_index_sequence<count>{});
        }

        return true;
    }
};

export template <dpp::simd_abi A, dpp::simd_algorithm_operation auto scanop>
class inclusive_scan {
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <dpp::simd_element_for<A> E, typename T>
    static constexpr T expected_op(T const& vals) noexcept {
        constexpr auto float_sum =
            dpl::floating_point_like<E> && scanop == dpp::exscan_sum;
        auto result = [] {
            using R = dpl::conditional_t<float_sum, long double, E>;
            if constexpr (dpp::fixed_width_abi<A>) {
                return array<R, abi_traits<E>::size>{};
            } else {
                return dynamic_array<R>(abi_traits<E>::size());
            }
        }();
        for (auto i = 0zu; i < vals.size(); ++i) {
            result[i] = vals[i];
        }

        for (auto i = 1zu; i < vals.size(); ++i) {
            auto const j = i - 1;
            if constexpr (scanop == dpp::scan_sum) {
                result[i] += result[j];
            } else if constexpr (scanop == dpp::scan_max) {
                result[i] = result[j] < result[i] ? result[i] : result[j];
            } else {
                result[i] = result[j] > result[i] ? result[i] : result[j];
            }
        }

        if constexpr (!float_sum) {
            return result;
        } else if constexpr (dpp::fixed_width_abi<A>) {
            auto output = array<E, abi_traits<E>::size>{};
            for (auto i = 0zu; i < vals.size(); ++i) {
                output[i] = result[i];
            }
            return output;
        } else {
            auto output = dynamic_array<E>(abi_traits<E>::size());
            for (auto i = 0zu; i < vals.size(); ++i) {
                output[i] = result[i];
            }
            return output;
        }
    }

    template <dpp::simd_element_for<A> E, typename T, typename M>
    static constexpr T expected_op(T const& vals, M mask) noexcept {
        auto clean(vals);
        for (auto i = 0zu; i < vals.size(); ++i) {
            if (!mask[i]) {
                clean[i] = scan_identity<scanop, E>();
            }
        }
        return expected_op<E>(clean);
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return inclusive_scan::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires dpp::fixed_width_abi<A>
    {
        constexpr auto width = abi_traits<E>::size();
        scanning_data_generator<scanop, A, E> const data_generator;
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
                auto const expected = expected_op<E>(val);
                auto const vexpected = dpp::load<A, E>(expected.data());
                auto const vactual = scanop(vval);
                if constexpr (dpl::is_integral_v<E>) {
                    assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                } else {
                    using sint_t = dpp::signed_representation_t<E>;
                    auto const diff = dpp::reinterpret<sint_t>(vactual) -
                        dpp::reinterpret<sint_t>(vexpected);
                    // less than 2 ulp
                    assert(dpp::all_of(dpp::cmplt(dpp::abs(diff), 2)));
                }
            }
            {
                auto const expected = expected_op<E>(val, mask);
                auto const vexpected = dpp::load<A, E>(expected.data());
                auto const vactual = scanop(vval, vmask);
                if constexpr (dpl::is_integral_v<E>) {
                    assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                } else {
                    using sint_t = dpp::signed_representation_t<E>;
                    auto const diff = dpp::reinterpret<sint_t>(vactual) -
                        dpp::reinterpret<sint_t>(vexpected);
                    // less than 2 ulp
                    assert(dpp::all_of(dpp::cmplt(dpp::abs(diff), 2)));
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
                        auto const expected = expected_op<E>(val, masks[I]);
                        auto const vexpected = dpp::load<A, E>(expected.data());
                        auto const vactual = scanop(vval, cmask);
                        if constexpr (dpl::is_integral_v<E>) {
                            assert(
                                dpp::all_of(test::bitcmp(vactual, vexpected)));
                        } else {
                            using sint_t = dpp::signed_representation_t<E>;
                            auto const diff =
                                dpp::reinterpret<sint_t>(vactual) -
                                dpp::reinterpret<sint_t>(vexpected);
                            // less than 2 ulp
                            assert(dpp::all_of(dpp::cmplt(dpp::abs(diff), 2)));
                        }
                    }
                },
                dpl::make_index_sequence<count>{});
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using scan_sum = inclusive_scan<A, dpp::scan_sum>;
export template <dpp::simd_abi A>
using scan_min = inclusive_scan<A, dpp::scan_min>;
export template <dpp::simd_abi A>
using scan_max = inclusive_scan<A, dpp::scan_max>;
export template <dpp::simd_abi A>
using exscan_sum = exclusive_scan<A, dpp::exscan_sum>;
export template <dpp::simd_abi A>
using exscan_min = exclusive_scan<A, dpp::exscan_min>;
export template <dpp::simd_abi A>
using exscan_max = exclusive_scan<A, dpp::exscan_max>;

export template <dpp::simd_abi A>
class exscan {
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return exscan::template run<E>(engine);
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

        scanning_data_generator<dpp::exscan_max, A, E> const max_generator;
        scanning_data_generator<dpp::exscan_min, A, E> const min_generator;

        for (auto i = 0; i < loop_count; ++i) {

            // Summation introduces error complexity, ignore it for now
            // This is anyway just a check for the generic scanning
            {
                auto const val = max_generator(engine);
                auto const init = max_generator.scalar(engine);
                auto const vval = dpp::load<A>(val.data());
                auto const expected = dpp::exscan_max(vval, init);
                auto const actual = dpp::exscan(vval, init, dpp::max);
                assert(dpp::all_of(test::bitcmp(actual, expected)));
            }
            {
                auto const val = min_generator(engine);
                auto const init = min_generator.scalar(engine);
                auto const vval = dpp::load<A>(val.data());
                auto const expected = dpp::exscan_min(vval, init);
                auto const actual = dpp::exscan(vval, init, dpp::min);
                assert(dpp::all_of(test::bitcmp(actual, expected)));
            }
        }

        return true;
    }
};

export template <dpp::simd_abi A>
class scan {
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return scan::template run<E>(engine);
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

        scanning_data_generator<dpp::scan_max, A, E> const max_generator;
        scanning_data_generator<dpp::scan_min, A, E> const min_generator;

        for (auto i = 0; i < loop_count; ++i) {

            // Summation introduces error complexity, ignore it for now
            // This is anyway just a check for the generic scanning
            {
                auto const val = max_generator(engine);
                auto const vval = dpp::load<A>(val.data());
                auto const expected = dpp::scan_max(vval);
                auto const actual = dpp::scan(vval, dpp::max);
                assert(dpp::all_of(test::bitcmp(actual, expected)));
            }
            {
                auto const val = min_generator(engine);
                auto const vval = dpp::load<A>(val.data());
                auto const expected = dpp::scan_min(vval);
                auto const actual = dpp::scan(vval, dpp::min);
                assert(dpp::all_of(test::bitcmp(actual, expected)));
            }
        }

        return true;
    }
};

} // namespace dpl::test
