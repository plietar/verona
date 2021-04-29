// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "bytecode/writer.h"
#include "compiler/codegen/descriptor.h"

namespace verona::compiler
{
  /**
   * Generate code for a builtin function.
   */
  void generate_builtin(
    BytecodeWriter& writer,
    ProgramTable& program_table,
    const CodegenItem<Method>& method);
}
