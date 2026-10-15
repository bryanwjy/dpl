// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.support:operation_fixture;
export import :comparison;

import dpl;

export namespace dpl::test {

namespace dpp = dpl::datapar;
inline namespace support {
// Generic test infrastructure for SIMD operations

struct broadcast_base {
protected:
    constexpr ~broadcast_base() = default;
};

template <typename T>
concept value_like = dpl::is_trivially_copyable_v<dpl::decay_t<T>>;

template <typename T>
struct splat : broadcast_base {
    using value_type = decay_t<T>;

    explicit constexpr splat(T&& arg) noexcept
    requires (!value_like<T>)
        : data(dpl::forward<T>(arg)) {}

    explicit constexpr splat(T arg) noexcept
    requires value_like<T>
        : data(dpl::move(arg)) {}

    constexpr ~splat() = default;

    constexpr T&& operator+() const noexcept { return dpl::forward<T>(data); }

    constexpr value_type operator+() const noexcept
    requires value_like<value_type>
    {
        return data;
    }

    using data_type = dpl::conditional_t<value_like<T>, value_type, T&&>;
    data_type data;
};

template <typename T>
requires (!value_like<T>)
explicit splat(T&& arg) -> splat<T>;

template <value_like T>
explicit splat(T arg) -> splat<T>;

template <typename T, typename A>
concept broadcastable_data_of = dpp::simd_abi<A> &&
    dpl::is_base_of_v<broadcast_base, T> && requires(T&& arg) {
        arg.~splat();
        dpp::broadcast<A>(+arg);
    };

template <typename T>
concept contiguous_data_base = requires(T& data) {
    typename T::value_type;
    typename T::reference;
    { data.data() } -> dpl::same_as<dpl::add_pointer_t<typename T::reference>>;
    { data.size() } -> dpl::unsigned_integral;
};

template <typename T>
concept contiguous_data = contiguous_data_base<remove_cvref_t<T>>;

template <typename T, typename A>
concept test_vector_type = dpp::simd_abi<A> &&
        (dpp::simd_vector<T> &&
            dpp::common_abi_with<dpp::simd_abi_type_t<T>, A>) ||
    contiguous_data<T> || (broadcastable_data_of<T, A> && requires(T&& arg) {
        { dpp::broadcast<A>(+arg) } -> dpp::simd_vector;
    });

template <typename T, typename A>
concept test_mask_type = dpp::simd_abi<A> &&
        (dpp::simd_mask<T> &&
            dpp::common_abi_with<dpp::simd_abi_type_t<T>, A>) ||
    dpp::const_mask_like<T> || dpl::bitset_type<T> ||
    (broadcastable_data_of<T, A> && requires(T&& arg) {
        { dpp::broadcast<A>(+arg) } -> dpp::simd_mask;
    });

template <typename T, typename A>
concept test_argument_type = test_vector_type<T, A> || test_mask_type<T, A>;

template <dpp::simd_abi A>
class operation_fixture {
    template <typename T>
    using value_t = typename T::value_type;

    template <typename T>
    static constexpr decltype(auto) load(T const& val) {
        if constexpr (broadcastable_data_of<T, A>) {
            return dpp::broadcast<A>(+val);
        } else if constexpr (contiguous_data<T>) {
            assert((val.size() == dpp::simd_abi_traits<A, value_t<T>>::size()));
            return dpp::load<A>(val.data());
        } else if constexpr (bitset_type<T>) {
            return dpp::from_bitset<A>(val);
        } else {
            return (val);
        }
    }

    template <typename T>
    using argument_t = decltype(operation_fixture::load(dpl::declval<T>()));

    struct success_t : dpl::true_type {
        constexpr success_t(bool val) noexcept { assert(val); }
    };

    template <typename T>
    static constexpr auto load_simd_mask(T mask) noexcept {
        if constexpr (dpp::const_mask_like<T>) {
            return dpp::from_bitset<A>(dpp::to_bitset(mask));
        } else {
            return operation_fixture::load(mask);
        }
    }

public:
    template <test_vector_type<A> T, typename... Ts,
        dpl::regular_invocable<argument_t<Ts>...> Op>
    static constexpr success_t test(
        T const& expected, Op op, Ts const&... args) {
        return test(dpp::cmpeq, expected, dpl::move(op), args...);
    }

    template <dpp::simd_element_for<A> T, typename... Ts,
        dpl::regular_invocable<argument_t<Ts>...> Op>
    static constexpr success_t test(
        T const& expected, Op op, Ts const&... args) {
        return test([](auto const& lhs, T const& rhs) { return lhs == rhs; },
            expected, dpl::move(op), args...);
    }

    template <test_mask_type<A> T, typename... Ts,
        dpl::regular_invocable<argument_t<Ts>...> Op>
    static constexpr success_t test(
        T const& expected, Op op, Ts const&... args) {
        auto const vexpected = operation_fixture::load(expected);
        dpp::simd_mask auto const vactual =
            op(operation_fixture::load(args)...);
        static_assert(requires {
            { dpp::cmpeq(vactual, vexpected) } -> dpp::simd_mask;
        });

        return dpp::all_of(dpp::cmpeq(vactual, expected));
    }

    template <test_vector_type<A> T, typename... Ts,
        dpl::regular_invocable<argument_t<Ts>...> Op,
        dpl::regular_invocable<dpl::invoke_result_t<Op, argument_t<Ts>...>,
            argument_t<T>>
            Cmp>
    static constexpr success_t test(
        Cmp cmp, T const& expected, Op op, Ts const&... args) {
        auto const vexpected = operation_fixture::load(expected);
        auto const vactual = op(operation_fixture::load(args)...);
        return dpp::all_of(cmp(vactual, vexpected));
    }

    template <dpp::simd_element_for<A> T, typename... Ts,
        dpl::regular_invocable<argument_t<Ts>...> Op,
        dpl::regular_invocable<dpl::invoke_result_t<Op, argument_t<Ts>...>, T>
            Cmp>
    static constexpr success_t test(
        Cmp cmp, T expected, Op op, Ts const&... args) {
        auto const actual = op(operation_fixture::load(args)...);
        return cmp(actual, expected);
    }

    template <typename Op, test_vector_type<A> S, test_mask_type<A> M,
        typename... Ts>
    static constexpr success_t test_masked(
        Op op, S const& src, M const& mask, Ts const&... args) {
        dpp::simd_vector auto const vsrc = operation_fixture::load(src);
        auto const mask_like = operation_fixture::load(mask);

        if constexpr (dpp::maskable_simd_accumulation<Op>) {
            static_assert(
                requires { op(vsrc, operation_fixture::load(args)...); });
        } else {
            static_assert(requires { op(operation_fixture::load(args)...); });
        }
        static_assert(requires {
            op(vsrc, mask_like, operation_fixture::load(args)...);
        });

        return [&](argument_t<Ts>... vargs) {
            dpp::simd_vector auto const vop = [&]() {
                if constexpr (dpp::maskable_simd_accumulation<Op>)
                    return op(vsrc, vargs...);
                else
                    return op(vargs...);
            }();
            dpp::simd_vector auto const vactual = op(vsrc, mask_like, vargs...);

            static_assert(dpl::is_same_v<decltype(vop), decltype(vactual)>);
            static_assert(dpl::is_same_v<decltype(vop), decltype(vsrc)>);

            auto const iftrue = test::bitcmp(vactual, vop);
            auto const iffalse = test::bitcmp(vactual, vsrc);

            return [&](dpp::simd_mask auto vmask) {
                auto const cond1 = dpp::cmpeq(dpp::bwand(iftrue, vmask), vmask);
                auto const cond2 =
                    dpp::cmpneq(dpp::bwandnot(iffalse, vmask), vmask);
                return dpp::all_of(dpp::logical_and(cond1, cond2));
            }(load_simd_mask(mask));
        }(operation_fixture::load(args)...);
    }

    template <typename Op, test_mask_type<A> M, typename... Ts>
    static constexpr success_t test_masked(
        Op op, dpp::zero_t zero, M const& mask, Ts const&... args) {
        static_assert(requires { op(operation_fixture::load(args)...); });
        static_assert(requires {
            op(zero, operation_fixture::load(mask),
                operation_fixture::load(args)...);
        });

        auto const mask_like = operation_fixture::load(mask);
        auto const vzero =
            dpp::broadcast<invoke_result_t<Op, argument_t<Ts>...>>(zero);

        return [&](argument_t<Ts>... vargs) {
            dpp::simd_vector auto const vop = op(vargs...);
            dpp::simd_vector auto const vactual = op(zero, mask_like, vargs...);
            dpp::simd_vector auto const valt = [&]() {
                if constexpr (dpp::maskable_simd_accumulation<Op>) {
                    return dpp::select(mask_like, op(vargs...), dpp::zero);
                } else {
                    return op(vzero, mask_like, vargs...);
                }
            }();

            static_assert(dpl::is_same_v<decltype(vop), decltype(vactual)>);
            static_assert(dpl::is_same_v<decltype(vop), decltype(valt)>);

            auto const iftrue = test::bitcmp(vactual, vop);
            auto const iffalse = test::bitcmp(vactual, vzero);

            return [&](dpp::simd_mask auto vmask) {
                dpp::simd_mask auto const cond1 =
                    dpp::cmpeq(dpp::bwand(iftrue, vmask), vmask);
                dpp::simd_mask auto const cond2 =
                    dpp::cmpneq(dpp::bwandnot(iffalse, vmask), vmask);
                dpp::simd_mask auto const cond3 = test::bitcmp(valt, vactual);
                assert(cond1.size() == cond2.size());
                assert(cond1.size() == cond3.size());
                return dpp::all_of(dpp::logical_and(cond1, cond2, cond3));
            }(load_simd_mask(mask));
        }(operation_fixture::load(args)...);
    }

    template <typename Op, test_mask_type<A> M, typename... Ts>
    static constexpr success_t test_masked(
        Op op, M const& mask, Ts const&... args) {
        static_assert(requires { op(operation_fixture::load(args)...); });
        static_assert(requires {
            op(operation_fixture::load(mask), operation_fixture::load(args)...);
        });

        auto const mask_like = operation_fixture::load(mask);
        return [&](argument_t<Ts>... vargs) {
            dpp::simd_mask auto const vop = op(vargs...);
            dpp::simd_mask auto const vactual = op(mask_like, vargs...);
            return dpp::all_of(dpp::cmpeq(
                vactual, dpp::logical_and(vop, load_simd_mask(mask))));
        }(operation_fixture::load(args)...);
    }
};
} // namespace support
} // namespace dpl::test
