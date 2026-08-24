// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: begin_exports
#if DPL_MODULES
#  include "dpl/xmm/basic/inl/compare.inl"
#endif

#include "dpl/xmm/operations/compare/cmpeq.h"
#include "dpl/xmm/operations/compare/cmpge.h"
#include "dpl/xmm/operations/compare/cmpgt.h"
#include "dpl/xmm/operations/compare/cmple.h"
#include "dpl/xmm/operations/compare/cmplt.h"
#include "dpl/xmm/operations/compare/cmpneq.h"
#include "dpl/xmm/operations/compare/cmpunord.h"
#include "dpl/xmm/operations/compare/max.h"
#include "dpl/xmm/operations/compare/min.h"
// IWYU pragma: end_exports
