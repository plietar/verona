// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::verona
{
  std::unique_ptr<Pass> createTypecheckerPass();

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "dialect/Passes.h.inc"
}
