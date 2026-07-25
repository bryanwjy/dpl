// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags: -fconstexpr-steps=10000000
// @dpl[msvc].compile-flags: /constexpr:steps10000000

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.operations.lane_broadcast;

// Tests for dpp::broadcast_lane on xmm ABI. Requires SSE4.2 (implied by
// dpl.xmm).
//
// Forms tested:
//   (1) dpp::broadcast_lane(vec, idx)                  -- unmasked, always
//   available (2) dpp::broadcast_lane(src, mask, vec, idx)       --
//   merge-masked (3) dpp::broadcast_lane(dpp::zero, mask, vec, idx) --
//   zero-masked explicit (4) dpp::broadcast_lane(mask, vec, idx)            --
//   zero-masked alias (== form 3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    using types = dpl::type_pack<dpl::int8, dpl::uint8, dpl::int16, dpl::uint16,
        dpl::int32, dpl::uint32, dpl::int64, dpl::uint64, float, double,
        dpl::ext::float16, dpl::ext::bfloat16>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::lane_broadcast<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
