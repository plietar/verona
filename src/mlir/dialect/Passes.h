// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::verona
{
  std::unique_ptr<Pass> createTypecheckerPass();
  std::unique_ptr<OperationPass<ModuleOp>> createPrintTopologicalFactsPass();

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "dialect/Passes.h.inc"
}
