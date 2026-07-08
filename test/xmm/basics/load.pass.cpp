// Generated with Claude
#include "../common.h"

import dpl.xmm;

namespace {
namespace xmm = dpl::datapar::xmm;
namespace dpp = dpl::datapar;
using abi_t = xmm::abi_tag;
template <typename E>
using vec_t = xmm::vector<E>;
template <typename E>
using abi_traits = dpp::simd_abi_traits<abi_t, E>;

template <typename T>
class DPL_EMPTY_BASES extended_stub :
    public vec_t<T>,
    public dpp::simd_vector_base<extended_stub<T>> {};

static_assert(dpp::simd_vector<extended_stub<int>>);

template <typename E>
void check_signatures() {
    using P = E const*;

    // all return the canonical vector type
    static_assert(dpl::is_same_v<decltype(dpp::load<abi_t>(dpl::declval<P>())),
        vec_t<E>>);
    static_assert(
        dpl::is_same_v<decltype(dpp::load<E, abi_t>(dpl::declval<P>())),
            vec_t<E>>);
    static_assert(
        dpl::is_same_v<decltype(dpp::load<abi_t, E>(dpl::declval<P>())),
            vec_t<E>>);

    // Order of element/abi does not matter for binary templates
    static_assert(
        dpl::is_same_v<decltype(dpp::load<E, abi_t>(dpl::declval<P>())),
            decltype(dpp::load<abi_t, E>(dpl::declval<P>()))>);

    // simd-vectors are decomposed into element and abi
    static_assert(
        dpl::is_same_v<decltype(dpp::load<vec_t<E>>(dpl::declval<P>())),
            vec_t<E>>);

    // load returns canonical even if template simd_vector is extended
    static_assert(
        dpl::is_same_v<decltype(dpp::load<extended_stub<E>>(dpl::declval<P>())),
            vec_t<E>>);
    static_assert(!dpl::is_same_v<decltype(dpp::load<extended_stub<E>>(
                                      dpl::declval<P>())),
        extended_stub<E>>);

    // All basic operations only ever involve canonical types and must be
    // noexcept
    static_assert(noexcept(dpp::load<abi_t>(dpl::declval<P>())));
    static_assert(noexcept(dpp::load<E, abi_t>(dpl::declval<P>())));
    static_assert(noexcept(dpp::load<abi_t, E>(dpl::declval<P>())));
    static_assert(noexcept(dpp::load<vec_t<E>>(dpl::declval<P>())));
    static_assert(noexcept(dpp::load<extended_stub<E>>(dpl::declval<P>())));

    static_assert(dpp::simd_basic_operation<dpp::load<abi_t>>);
    static_assert(dpp::simd_basic_operation<dpp::load<E, abi_t>>);
    static_assert(dpp::simd_basic_operation<dpp::load<abi_t, E>>);
    static_assert(dpp::simd_basic_operation<dpp::load<vec_t<E>>>);
    static_assert(dpp::simd_basic_operation<dpp::load<extended_stub<E>>>);

    static_assert(dpp::simd_basic_operation<dpp::load<abi_t>>);
    static_assert(dpp::simd_basic_operation<dpp::load<E, abi_t>>);
    static_assert(dpp::simd_basic_operation<dpp::load<abi_t, E>>);
    static_assert(dpp::simd_basic_operation<dpp::load<vec_t<E>>>);
    static_assert(dpp::simd_basic_operation<dpp::load<extended_stub<E>>>);

    static_assert(dpp::simd_canonical_invocable<dpp::load<abi_t>, E const*>);
    static_assert(dpp::simd_canonical_invocable<dpp::load<E, abi_t>, E const*>);
    static_assert(dpp::simd_canonical_invocable<dpp::load<abi_t, E>, E const*>);
    static_assert(dpp::simd_canonical_invocable<dpp::load<vec_t<E>>, E const*>);
    static_assert(
        dpp::simd_canonical_invocable<dpp::load<extended_stub<E>>, E const*>);
    static_assert(dpp::simd_canonical_invocable<dpp::load<abi_t>, E const*>);
    static_assert(dpp::simd_canonical_invocable<dpp::load<E, abi_t>,
        dpp::aligned_t, E const*>);
    static_assert(dpp::simd_canonical_invocable<dpp::load<abi_t, E>,
        dpp::aligned_t, E const*>);
    static_assert(dpp::simd_canonical_invocable<dpp::load<vec_t<E>>,
        dpp::aligned_t, E const*>);
    static_assert(dpp::simd_canonical_invocable<dpp::load<extended_stub<E>>,
        dpp::aligned_t, E const*>);
}

void check_signatures() {
    check_signatures<float>();
    check_signatures<double>();
    check_signatures<dpl::int8>();
    check_signatures<dpl::uint8>();
    check_signatures<dpl::int16>();
    check_signatures<dpl::uint16>();
    check_signatures<dpl::int32>();
    check_signatures<dpl::uint32>();
    check_signatures<dpl::int64>();
    check_signatures<dpl::uint64>();
#if DPL_SUPPORTS_FLOAT16
    check_signatures<dpl::float16>();
#endif
#if DPL_SUPPORTS_BFLOAT16
    check_signatures<dpl::float16>();
#endif
}

// Verifies a loaded vector round-trips through store() back to the source
// values.
template <typename E>
constexpr void expect_equal(
    vec_t<E> const& v, E const (&expected)[abi_traits<E>::size]) {

    E out[abi_traits<E>::size]{};
    dpp::store(v, out);
    for (auto i = 0zu; i < abi_traits<E>::size; ++i) {
        assert(out[i] == expected[i]);
    }
}

template <typename E>
constexpr void test_element() {
    constexpr dpl::size_t lanes = abi_traits<E>::size;
    using array_t = E[lanes];

    // Exact-size buffer (no padding either side): under ASan this turns any
    // over-read by load() into a reliable failure rather than an unspecified
    // pass.
    alignas(typename abi_traits<E>::native_vector) array_t in{};
    for (auto i = 0zu; i < lanes; ++i) {
        in[i] = static_cast<E>(i + 1); // 1..lanes: nonzero, distinct per lane
    }

    {
        // (1) A explicit, E deduced
        auto v1 = dpp::load<abi_t>(in);
        static_assert(dpl::is_same_v<decltype(v1), vec_t<E>>);
        expect_equal(v1, in);

        // (2) E, A both explicit
        auto v2 = dpp::load<E, abi_t>(in);
        static_assert(dpl::is_same_v<decltype(v2), vec_t<E>>);
        expect_equal(v2, in);

        // (3) A, E reversed -- must behave identically to (2)
        auto v3 = dpp::load<abi_t, E>(in);
        static_assert(dpl::is_same_v<decltype(v3), vec_t<E>>);
        expect_equal(v3, in);

        // (4) T = canonical vector type, redirects to (2)
        auto v4 = dpp::load<vec_t<E>>(in);
        static_assert(dpl::is_same_v<decltype(v4), vec_t<E>>);
        expect_equal(v4, in);
    }

    // Aligned
    {
        // (1) A explicit, E deduced
        auto v1 = dpp::load<abi_t>(dpp::aligned, in);
        static_assert(dpl::is_same_v<decltype(v1), vec_t<E>>);
        expect_equal(v1, in);

        // (2) E, A both explicit
        auto v2 = dpp::load<E, abi_t>(dpp::aligned, in);
        static_assert(dpl::is_same_v<decltype(v2), vec_t<E>>);
        expect_equal(v2, in);

        // (3) A, E reversed -- must behave identically to (2)
        auto v3 = dpp::load<abi_t, E>(dpp::aligned, in);
        static_assert(dpl::is_same_v<decltype(v3), vec_t<E>>);
        expect_equal(v3, in);

        // (4) T = canonical vector type, redirects to (2)
        auto v4 = dpp::load<vec_t<E>>(dpp::aligned, in);
        static_assert(dpl::is_same_v<decltype(v4), vec_t<E>>);
        expect_equal(v4, in);
    }
}

// (4) again, but with T = an *extended* type. Return type must still be
// canonical -- this is the behavior the spec calls out explicitly.
template <typename E>
constexpr void test_extended_redirect() {
    constexpr dpl::size_t lanes = abi_traits<E>::size;
    using array_t = E[lanes];

    array_t in{};
    for (auto i = 0zu; i < lanes; ++i) {
        in[i] = static_cast<E>(i + 1);
    }

    {
        auto v = dpp::load<extended_stub<E>>(in);
        static_assert(dpl::is_same_v<decltype(v), vec_t<E>>,
            "load<T> must return a canonical basic_vector even when "
            "T is extended");
        expect_equal(v, in);
    }

    {
        auto v = dpp::load<extended_stub<E>>(dpp::aligned, in);
        static_assert(dpl::is_same_v<decltype(v), vec_t<E>>,
            "load<T> must return a canonical basic_vector even when "
            "T is extended");
        expect_equal(v, in);
    }
}

constexpr bool test_all() {
    test_element<float>();
    test_element<double>();
    test_element<dpl::int8>();
    test_element<dpl::uint8>();
    test_element<dpl::int16>();
    test_element<dpl::uint16>();
    test_element<dpl::int32>();
    test_element<dpl::uint32>();
    test_element<dpl::int64>();
    test_element<dpl::uint64>();

    test_extended_redirect<float>();
    test_extended_redirect<dpl::int32>();
    return true;
}

} // namespace

int main() {
    static_assert(test_all());
    assert(test_all());
    return 0;
}
