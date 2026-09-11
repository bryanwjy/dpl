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
        } else {
            auto output = test::make_array<E, A>();
            output[0] = init;
            for (auto const i : linear_counter(output)) {
                output[i] = result[i];
            }
            return output;
        }
    }

    template <dpp::simd_element_for<A> E, typename T, typename M>
    static constexpr T expected_op(T const& vals, M mask, E init) noexcept {
        auto clean(vals);
        auto const amask = [&] {
            if constexpr (dpp::const_mask_like<M>) {
                return mask;
            } else {
                return test::to_mask_array<E, A>(mask);
            }
        }();
        for (auto const i : linear_counter(vals)) {
            if (amask[i] == 0) {
                clean[i] = scan_identity<scanop, E>();
            }
        }
        return expected_op(clean, init);
    }

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = dpp::simd_abi_traits<A, E>::size();
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
        return linear_counter(max < limit ? max : limit);
    }

    template <typename E>
    static constexpr void run_const_mask_test(auto func) noexcept {
        [&]<size_t I = 0, serialized_mt19937 S = {}>(this auto self,
            dpp::immediate<I> = dpp::imm<I>, mt19937_type<S> = {}) {
            if constexpr (I < test_count<E>().size()) {
                constexpr auto lanes = dpp::simd_abi_traits<A, E>::size();
                constexpr bit_generator<lanes> bitgen;
                constexpr auto pair = S.generate_with(bitgen);
                constexpr auto cmask = dpp::deduce_const_mask_v<pair.value>;

                if constexpr (dpp::none_of(cmask)) {
                    self(dpp::imm<I>, mt19937_type<pair.state>{});
                } else {
                    func(cmask);
                    self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
                }
            }
        }
        ();
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
    static constexpr bool run(Rng& engine) {
        scanning_data_generator<scanop, A, E> const data_generator;
        mask_generator<A, E> const mask_generator;

        for (auto const _ : test_count<E>()) {
            auto const val = data_generator(engine);
            auto const init = data_generator.scalar(engine);

            {
                auto const expected = expected_op(val, init);
                operation_fixture<A>::test(
                    precision_cmp<3>, expected, scanop, val, init);
            }
            {
                auto const mask = [&] {
                    auto mask = mask_generator(engine);
                    while (dpp::none_of(mask)) {
                        mask = mask_generator(engine);
                    }
                    return mask;
                }();
                auto const expected = expected_op(val, mask, init);
                operation_fixture<A>::test(
                    precision_cmp<3>, expected, scanop, val, mask, init);
            }
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const val = data_generator(engine);
            auto const init = data_generator.scalar(engine);
            run_const_mask_test<E>([&](auto cmask) {
                auto const expected = expected_op(val, cmask, init);
                operation_fixture<A>::test(
                    precision_cmp<3>, expected, scanop, val, cmask, init);
            });
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
        auto const amask = [&] {
            if constexpr (dpp::const_mask_like<M>) {
                return mask;
            } else {
                return test::to_mask_array<E, A>(mask);
            }
        }();
        for (auto const i : linear_counter(vals)) {
            if (amask[i] == 0) {
                clean[i] = scan_identity<scanop, E>();
            }
        }
        return expected_op<E>(clean);
    }

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = dpp::simd_abi_traits<A, E>::size();
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
        return linear_counter(max < limit ? max : limit);
    }

    template <typename E>
    static constexpr void run_const_mask_test(auto func) noexcept {
        [&]<size_t I = 0, serialized_mt19937 S = {}>(this auto self,
            dpp::immediate<I> = dpp::imm<I>, mt19937_type<S> = {}) {
            if constexpr (I < test_count<E>().size()) {
                constexpr auto lanes = dpp::simd_abi_traits<A, E>::size();
                constexpr bit_generator<lanes> bitgen;
                constexpr auto pair = S.generate_with(bitgen);
                constexpr auto cmask = dpp::deduce_const_mask_v<pair.value>;

                if constexpr (dpp::none_of(cmask)) {
                    self(dpp::imm<I>, mt19937_type<pair.state>{});
                } else {
                    func(cmask);
                    self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
                }
            }
        }
        ();
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
    static constexpr bool run(Rng& engine) {
        constexpr auto width = abi_traits<E>::size();
        scanning_data_generator<scanop, A, E> const data_generator;
        mask_generator<A, E> const mask_generator;

        for (auto const _ : test_count<E>()) {
            auto const val = data_generator(engine);
            {
                auto const expected = expected_op<E>(val);
                operation_fixture<A>::test(
                    precision_cmp<3>, expected, scanop, val);
            }
            {
                auto const mask = [&] {
                    auto mask = mask_generator(engine);
                    while (dpp::none_of(mask)) {
                        mask = mask_generator(engine);
                    }
                    return mask;
                }();
                auto const expected = expected_op<E>(val, mask);
                operation_fixture<A>::test(
                    precision_cmp<3>, expected, scanop, val, mask);
            }
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const val = data_generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                auto const expected = expected_op<E>(val, cmask);
                operation_fixture<A>::test(
                    precision_cmp<3>, expected, scanop, val, cmask);
            });
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
    template <typename E>
    static constexpr linear_counter test_count() noexcept {
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
        return linear_counter(max < limit ? max : limit);
    }

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
        scanning_data_generator<dpp::exscan_max, A, E> const max_generator;
        scanning_data_generator<dpp::exscan_min, A, E> const min_generator;
        test::scalar_generator<E> const init_generator;

        for (auto const _ : test_count<E>()) {
            auto const init = init_generator(engine);
            {
                auto const val = dpp::load<A>(max_generator(engine).data());
                auto const expected = dpp::exscan_max(val, init);
                test::operation_fixture<A>::test(precision_cmp<3>, expected,
                    dpp::exscan, val, init, dpp::max);
            }
            {
                auto const val = dpp::load<A>(min_generator(engine).data());
                auto const expected = dpp::exscan_min(val, init);
                test::operation_fixture<A>::test(precision_cmp<3>, expected,
                    dpp::exscan, val, init, dpp::min);
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
    template <typename E>
    static constexpr linear_counter test_count() noexcept {
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
        return linear_counter(max < limit ? max : limit);
    }

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
        scanning_data_generator<dpp::exscan_max, A, E> const max_generator;
        scanning_data_generator<dpp::exscan_min, A, E> const min_generator;

        for (auto const _ : test_count<E>()) {
            {
                auto const val = dpp::load<A>(max_generator(engine).data());
                auto const expected = dpp::scan_max(val);
                test::operation_fixture<A>::test(
                    precision_cmp<3>, expected, dpp::scan, val, dpp::max);
            }
            {
                auto const val = dpp::load<A>(min_generator(engine).data());
                auto const expected = dpp::scan_min(val);
                test::operation_fixture<A>::test(
                    precision_cmp<3>, expected, dpp::scan, val, dpp::min);
            }
        }

        return true;
    }
};

} // namespace dpl::test
