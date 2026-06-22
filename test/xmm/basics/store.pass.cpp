// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;

namespace {
namespace xmm = dpl::datapar::xmm;
namespace dpp = dpl::datapar;
using abi_t = xmm::abi_tag;
template <typename E>
using vec_t = xmm::simd<E>;
template <typename E>
using abi_traits = dpp::simd_abi_traits<abi_t, E>;

template <typename E>
void check_signature() {
    using P = E*;
    using V = vec_t<E> const&;

    static_assert(dpl::is_same_v<
        decltype(dpp::store(dpl::declval<V>(), dpl::declval<P>())), void>);
    static_assert(noexcept(dpp::store(dpl::declval<V>(), dpl::declval<P>())));
}

void check_signatures() {
    check_signature<float>();
    check_signature<double>();
    check_signature<dpl::int8>();
    check_signature<dpl::uint8>();
    check_signature<dpl::int16>();
    check_signature<dpl::uint16>();
    check_signature<dpl::int32>();
    check_signature<dpl::uint32>();
    check_signature<dpl::int64>();
    check_signature<dpl::uint64>();
#if DPL_SUPPORTS_FLOAT16
    check_signature<dpl::float16>();
#endif
#if DPL_SUPPORTS_BFLOAT16
    check_signature<dpl::float16>();
#endif
}

template <typename E>
constexpr void test_element() {
    constexpr auto lanes = abi_traits<E>::size;

    E in[lanes]{};
    for (auto i = 0zu; i < lanes; ++i) {
        in[i] = static_cast<E>(i + 1); // 1..lanes: nonzero, distinct per lane
    }
    auto v = dpp::load<abi_t>(in);

    // (1) exact-size destination: every lane lands in the right place.
    E out[lanes]{};
    dpp::store(v, out);
    for (auto i = 0zu; i < lanes; ++i) {
        assert(out[i] == in[i]);
    }

    // (2) store into the middle of a larger buffer: confirms store(v, ptr)
    // writes exactly `lanes` contiguous elements starting at ptr, touching
    // nothing on either side. An over/under-write either fails to be a
    // constant expression (caught by static_assert) or, at runtime under
    // ASan, is a reliable failure.
    constexpr E sentinel = E{0}; // distinct from in's 1..lanes range
    E guarded[lanes + 2]{};
    guarded[0] = sentinel;
    guarded[lanes + 1] = sentinel;

    dpp::store(v, guarded + 1);

    assert(guarded[0] == sentinel);
    assert(guarded[lanes + 1] == sentinel);
    for (auto i = 0zu; i < lanes; ++i) {
        assert(guarded[i + 1] == in[i]);
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
    return true;
}

} // namespace

int main() {
    static_assert(test_all());
    assert(test_all());
    return 0;
}
