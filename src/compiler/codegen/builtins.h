// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "bytecode/generator.h"
#include "compiler/codegen/descriptor.h"

namespace verona::compiler
{
  /**
   * Generate code for a builtin function.
   */
  void generate_builtin(
    Generator& writer,
    ProgramTable& program_table,
    const CodegenItem<Method>& method);
}
