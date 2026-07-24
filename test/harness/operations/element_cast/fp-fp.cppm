// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>
export module dpl.test.harness.operations.element_cast:fp_fp;
export import dpl.test.support;

import dpl;

export namespace dpl::test {

namespace dpp = dpl::datapar;

template <dpp::simd_abi A>
class fp_fp_cast {
private:
    using abi_t = A;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<abi_t, E>;

    template <dpl::floating_point_like To, dpl::floating_point_like From,
        typename Pred = decltype(dpp::cmpeq)>
    requires dpl::different_from<From, To>
    static constexpr bool round_trip(
        From val, Pred pred = dpp::cmpeq) noexcept {
        constexpr auto count =
            dpp::min(abi_traits<From>::size(), abi_traits<To>::size());
        constexpr auto M = (1 << count) - 1;
        auto const src =
            dpp::selecti<M>(dpp::broadcast<From, abi_t>(val), dpp::zero);
        static_assert(dpl::same_as<From,
            typename dpl::remove_const_t<decltype(src)>::value_type>);
        auto const dst = dpp::selecti<M>(
            dpp::broadcast<To, abi_t>(static_cast<To>(val)), dpp::zero);
        auto const actual_dst = dpp::element_cast<To>(src);
        auto const dstequal = pred(actual_dst, dst);
        auto const actual_castback = dpp::element_cast<From>(actual_dst);
        auto const isequal = pred(actual_castback, src);
        return dpp::all_of(isequal) && dpp::all_of(dstequal);
    }

    static constexpr auto nextafter(auto val) noexcept {
        using type = decltype(val);
        using rep = dpp::signed_representation_t<type>;
        return dpl::bit_cast<type>(
            static_cast<rep>(dpl::bit_cast<rep>(val) + 1));
    }

    static constexpr auto nextbefore(auto val) noexcept {
        using type = decltype(val);
        using rep = dpp::signed_representation_t<type>;
        return dpl::bit_cast<type>(
            static_cast<rep>(dpl::bit_cast<rep>(val) - 1));
    }

    template <dpl::floating_point_like To, dpl::floating_point_like From,
        typename Pred = decltype(dpp::cmpeq)>
    requires (dpl::floating_point_traits<To>::digits !=
        dpl::floating_point_traits<From>::digits)
    static constexpr bool one_way(From val, Pred pred = dpp::cmpeq) noexcept {
        constexpr auto count =
            dpp::min(abi_traits<From>::size(), abi_traits<To>::size());
        constexpr auto M = (1 << count) - 1;
        auto const src =
            dpp::selecti<M>(dpp::broadcast<From, abi_t>(val), dpp::zero);
        auto const dst = dpp::selecti<M>(
            dpp::broadcast<To, abi_t>(static_cast<To>(val)), dpp::zero);
        auto const actual_dst = dpp::element_cast<To>(src);
        auto const dstequal = pred(actual_dst, dst);
        auto const actual_castback = dpp::element_cast<From>(actual_dst);
        auto const isequal = pred(actual_castback, src);
        return dpp::all_of(dstequal) && dpp::none_of(isequal);
    }

    static constexpr void test() {
        assert(round_trip<double>(0.0f));
        assert(round_trip<double>(-0.0f, test::bitcmp));
        assert(round_trip<double>(1.0f));
        assert(round_trip<double>(nextafter(1.0f)));
        assert(round_trip<double>(nextbefore(1.0f)));
        assert(round_trip<double>(0x1.p24f));
        assert(round_trip<double>(dpl::bit_cast<float>(1 << 23)));
        assert(round_trip<double>(dpl::bit_cast<float>(1)));
        assert(round_trip<double>(1e-40f));
        assert(round_trip<double>(dpp::max_value_v<float>));
        assert(round_trip<double>(dpp::infinity_v<float>));
        assert(round_trip<double>(-dpp::infinity_v<float>));

        // qnan
        assert(
            round_trip<double>(dpl::bit_cast<float>(0x7fc00000), test::nancmp));
        // snan
        assert(
            round_trip<double>(dpl::bit_cast<float>(0x7fa00000), test::nancmp));

        assert(round_trip<float>(0.0));
        assert(round_trip<float>(-0.0, test::bitcmp));
        assert(round_trip<float>(1.0));
        assert(round_trip<float>(0x1.p24));
        assert(round_trip<float>(dpp::infinity_v<double>));
        assert(round_trip<float>(-dpp::infinity_v<double>));
        // qnan
        assert(round_trip<float>(
            dpl::bit_cast<double>(0x7ff8ll << 48), test::nancmp));
        // snan
        assert(round_trip<float>(
            dpl::bit_cast<double>(0x7ff4ll << 48), test::nancmp));

        assert(one_way<float>(nextafter(1.0)));
        assert(one_way<float>(nextbefore(1.0)));
        assert(one_way<float>(dpl::bit_cast<double>(1ll << 52)));
        assert(one_way<float>(dpl::bit_cast<double>(1ll)));

        // Three values where there is insufficient precision
        assert(one_way<float>(0x1.p24 + 1.0));
        assert(one_way<float>(0x1.p10 + 0x1.p-14));
        assert(one_way<float>(1.0 + 0x1.p-24));
    }

    static constexpr void test_fp16() noexcept {
        using bfloat16 = dpl::ext::bfloat16;
        using float16 = dpl::ext::float16;
        using dpl::ext_literals::operator""_f16;
        assert(round_trip<float>(0.0_f16));
        assert(round_trip<float>(-0.0_f16, test::bitcmp));
        assert(round_trip<float>(1.0_f16));
        assert(round_trip<float>(nextafter(1.0_f16)));
        assert(round_trip<float>(nextbefore(1.0_f16)));
        assert(round_trip<float>(0x1.p11_f16));
        assert(round_trip<float>(
            dpl::bit_cast<float16>(static_cast<short>(1 << 11))));
        assert(
            round_trip<float>(dpl::bit_cast<float16>(static_cast<short>(1))));
        assert(round_trip<float>(1e-6_f16));
        assert(round_trip<float>(dpp::max_value_v<float16>));
        assert(round_trip<float>(dpp::infinity_v<float16>));
        assert(round_trip<float>(-dpp::infinity_v<float16>));
        assert(one_way<float16>(nextafter(1.0f)));
        assert(one_way<float16>(nextbefore(1.0f)));
        assert(one_way<float16>(dpl::bit_cast<float>(1 << 23)));
        assert(one_way<float16>(dpl::bit_cast<float>(1)));
        assert(one_way<float16>(0x1.p11f + 1.0f));
        assert(one_way<float16>(0x1.p5f + 0x1.p-6f));
        assert(one_way<float16>(1.0f + 0x1.p-11f));

        // qnan
        assert(round_trip<float>(
            dpl::bit_cast<float16>(static_cast<short>(0x7e00)), test::nancmp));
        // snan
        assert(round_trip<float>(
            dpl::bit_cast<float16>(static_cast<short>(0x7c01)), test::nancmp));

        assert(round_trip<double>(0.0_f16));
        assert(round_trip<double>(-0.0_f16, test::bitcmp));
        assert(round_trip<double>(1.0_f16));
        assert(round_trip<double>(nextafter(1.0_f16)));
        assert(round_trip<double>(nextbefore(1.0_f16)));
        assert(round_trip<double>(0x1.p11_f16));
        assert(round_trip<double>(
            dpl::bit_cast<float16>(static_cast<short>(1 << 10))));
        assert(
            round_trip<double>(dpl::bit_cast<float16>(static_cast<short>(1))));
        assert(round_trip<double>(1e-6_f16));
        assert(round_trip<double>(dpp::max_value_v<float16>));
        assert(round_trip<double>(dpp::infinity_v<float16>));
        assert(round_trip<double>(-dpp::infinity_v<float16>));
        // qnan
        assert(round_trip<double>(
            dpl::bit_cast<float16>(static_cast<short>(0x7e00)), test::nancmp));
        // snan
        assert(round_trip<double>(
            dpl::bit_cast<float16>(static_cast<short>(0x7c01)), test::nancmp));

        assert(one_way<float16>(nextafter(1.0)));
        assert(one_way<float16>(nextbefore(1.0)));
        assert(one_way<float16>(dpl::bit_cast<double>(1ll << 52)));
        assert(one_way<float16>(dpl::bit_cast<double>(1ll)));
        assert(one_way<float16>(0x1.p11 + 1.0));
        assert(one_way<float16>(0x1.p5 + 0x1.p-6));
        assert(one_way<float16>(1.0 + 0x1.p-11));

        assert(round_trip<bfloat16>(0.0_f16));
        assert(round_trip<bfloat16>(-0.0_f16, test::bitcmp));
        assert(round_trip<bfloat16>(1.0_f16));
        assert(one_way<bfloat16>(nextafter(1.0_f16)));
        assert(one_way<bfloat16>(nextbefore(1.0_f16)));
        assert(round_trip<bfloat16>(0x1.p8_f16));
        assert(round_trip<bfloat16>(dpp::infinity_v<float16>));
        assert(round_trip<bfloat16>(-dpp::infinity_v<float16>));
        assert(round_trip<bfloat16>(1e-6_f16));
        assert(one_way<bfloat16>(dpp::max_value_v<float16>));
        assert(round_trip<bfloat16>(
            dpl::bit_cast<float16>(static_cast<unsigned short>(1 << 10))));
        assert(round_trip<bfloat16>(
            dpl::bit_cast<float16>(static_cast<unsigned short>(1))));
    }

    static constexpr void test_bf16() noexcept {
        using dpl::ext_literals::operator""_bf16;
        using bfloat16 = dpl::ext::bfloat16;
        using float16 = dpl::ext::float16;

        constexpr auto min_normal =
            static_cast<bfloat16>(dpl::bit_cast<float>(1 << 23));
        constexpr auto min_subnormal =
            static_cast<bfloat16>(dpl::bit_cast<float>(1 << 16));

        assert(round_trip<float>(0.0_bf16));
        assert(round_trip<float>(-0.0_bf16, test::bitcmp));
        assert(round_trip<float>(1.0_bf16));
        assert(round_trip<float>(nextafter(1.0_bf16)));
        assert(round_trip<float>(nextbefore(1.0_bf16)));
        assert(round_trip<float>(0x1.p8_bf16));
        assert(round_trip<float>(1e-40_bf16));
        assert(round_trip<float>(dpp::max_value_v<bfloat16>));
        assert(round_trip<float>(dpp::infinity_v<bfloat16>));
        assert(round_trip<float>(-dpp::infinity_v<bfloat16>));
        assert(round_trip<float>(min_normal));
        assert(round_trip<float>(min_subnormal));
        assert(one_way<bfloat16>(nextafter(1.0f)));
        assert(one_way<bfloat16>(nextbefore(1.0f)));
        assert(one_way<bfloat16>(dpl::bit_cast<float>(1)));
        assert(one_way<bfloat16>(0x1.p8f + 1.0f));
        assert(one_way<bfloat16>(0x1.p4f + 0x1.p-4f));
        assert(one_way<bfloat16>(1.0f + 0x1.p-8f));

        assert(round_trip<double>(0.0_bf16));
        assert(round_trip<double>(-0.0_bf16, test::bitcmp));
        assert(round_trip<double>(1.0_bf16));
        assert(round_trip<double>(nextafter(1.0_bf16)));
        assert(round_trip<double>(nextbefore(1.0_bf16)));
        assert(round_trip<double>(0x1.p8_bf16));
        assert(round_trip<double>(1e-40_bf16));
        assert(round_trip<double>(dpp::max_value_v<bfloat16>));
        assert(round_trip<double>(dpp::infinity_v<bfloat16>));
        assert(round_trip<double>(-dpp::infinity_v<bfloat16>));
        assert(round_trip<double>(min_normal));
        assert(round_trip<double>(min_subnormal));
        assert(one_way<bfloat16>(nextafter(1.0)));
        assert(one_way<bfloat16>(nextbefore(1.0)));
        assert(one_way<bfloat16>(dpl::bit_cast<double>(1ll << 52)));
        assert(one_way<bfloat16>(dpl::bit_cast<double>(1ll)));
        assert(one_way<bfloat16>(0x1.p8 + 1.0));
        assert(one_way<bfloat16>(0x1.p4 + 0x1.p-4));
        assert(one_way<bfloat16>(1.0 + 0x1.p-8));
        assert(round_trip<float16>(0.0_bf16));
        assert(round_trip<float16>(-0.0_bf16, test::bitcmp));
        assert(round_trip<float16>(1.0_bf16));
        assert(round_trip<float16>(nextafter(1.0_bf16)));
        assert(round_trip<float16>(nextbefore(1.0_bf16)));
        assert(round_trip<float16>(0x1.p8_bf16));
        assert(round_trip<float16>(dpp::infinity_v<bfloat16>));
        assert(round_trip<float16>(-dpp::infinity_v<bfloat16>));
        assert(one_way<float16>(1e-40_bf16));
        assert(one_way<float16>(dpp::max_value_v<bfloat16>));
        assert(one_way<float16>(min_normal));
        assert(one_way<float16>(min_subnormal));
    }

public:
    static constexpr bool run_all() {
        test();
        test_bf16();
        test_fp16();
        return true;
    }
};

} // namespace dpl::test
