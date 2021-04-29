// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/analysis.h"
#include "compiler/codegen/descriptor.h"
#include "compiler/ir/ir.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  void generate_ir_function(
    Context& context,
    ProgramTable& program_table,
    const SelectorTable& selectors,
    bytecode::BytecodeWriter& writer,
    const CodegenItem<Method>& method,
    const FnAnalysis& analysis);
}
