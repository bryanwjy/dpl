// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.xmm:operations;
import :details.operations;
import :basic;

// NOLINTBEGIN(misc-unused-using-decls)
__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {
using __DPL datapar::xmm::abs;
using __DPL datapar::xmm::add;
using __DPL datapar::xmm::divide;
using __DPL datapar::xmm::multiply;
// using __DPL datapar::xmm::negate;
using __DPL datapar::xmm::subadd;
using __DPL datapar::xmm::subtract;
//
using __DPL datapar::xmm::bwand;
using __DPL datapar::xmm::bwandnot;
using __DPL datapar::xmm::bwnot;
using __DPL datapar::xmm::bwor;
using __DPL datapar::xmm::bwornot;
using __DPL datapar::xmm::bwshift_left;
using __DPL datapar::xmm::bwshift_right;
using __DPL datapar::xmm::bwxor;
//
using __DPL datapar::xmm::byteswap;
using __DPL datapar::xmm::countl_one;
using __DPL datapar::xmm::countl_zero;
using __DPL datapar::xmm::countr_one;
using __DPL datapar::xmm::countr_zero;
using __DPL datapar::xmm::popcount;
// using __DPL datapar::xmm::rotl;
// using __DPL datapar::xmm::rotr;
//

//
using __DPL datapar::xmm::cmpeq;
using __DPL datapar::xmm::cmpge;
using __DPL datapar::xmm::cmpgt;
using __DPL datapar::xmm::cmple;
using __DPL datapar::xmm::cmplt;
using __DPL datapar::xmm::cmpneq;
using __DPL datapar::xmm::cmpunord;
// using __DPL datapar::xmm::min;
// using __DPL datapar::xmm::max;
//
using __DPL datapar::xmm::element_cast;
//
using __DPL datapar::xmm::reinterpret;
using __DPL datapar::xmm::select;
using __DPL datapar::xmm::to_simd_mask;
} // namespace datapar::xmm
__DPL_DEFAULT_NAMESPACE_END
// NOLINTEND(misc-unused-using-decls)
