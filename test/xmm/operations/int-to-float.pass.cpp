

import dpl.xmm;

namespace xmm = dpl::datapar::xmm;
namespace dpp = dpl::datapar;

static_assert(dpl::datapar::simd_type<xmm::simd<float>>);

constexpr auto val = dpp::broadcast<long, xmm::abi_tag>(17ll);

static_assert(
    (dpp::cast<float>(val) == xmm::simd<float>(17.0f, 17.0f, 0.0f, 0.0f))[0]);
int main() {
    auto const data = dpp::broadcast<float, xmm::abi_tag>(0.0f);
    return 0;
}
