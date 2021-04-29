// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "bytecode/generator.h"
#include "bytecode/writer.h"
#include "compiler/analysis.h"
#include "compiler/codegen/descriptor.h"

namespace verona::compiler
{
  void emit_functions(
    Context& context,
    const AnalysisResults& analysis,
    const Reachability& reachability,
    ProgramTable& program_table,
    const SelectorTable& selectors,
    BytecodeWriter& writer);
}
