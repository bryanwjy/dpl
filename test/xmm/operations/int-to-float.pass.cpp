

import dpl.xmm;

namespace xmm = dpl::datapar::xmm;

static_assert(
    dpl::datapar::simd_type<dpl::datapar::basic_simd<float, xmm::abi_tag>>);

int main() {
    auto const data = dpl::datapar::broadcast<float, xmm::abi_tag>(0.0f);
    return 0;
}
