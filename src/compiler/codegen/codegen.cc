// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/codegen.h"

#include "bytecode/bytecode.h"
#include "bytecode/writer.h"
#include "compiler/ast.h"
#include "compiler/codegen/descriptor.h"
#include "compiler/codegen/function.h"
#include "compiler/reachability/reachability.h"

namespace verona::compiler
{
  /**
   * Writes the magic numbers to the bytecode
   * @param code BytecodeWriter to which the bytes should be emitted
   */
  void write_magic_number(BytecodeWriter& code)
  {
    code.u32(bytecode::MAGIC_NUMBER);
  }

  std::vector<uint8_t> codegen(
    Context& context, const Program& program, const AnalysisResults& analysis)
  {
    auto entry = find_program_entry(context, program);
    if (!entry)
      return {};

    std::vector<uint8_t> code;

    BytecodeWriter writer(code);
    write_magic_number(writer);

    Reachability reachability =
      compute_reachability(context, program, *entry, analysis);
    SelectorTable selectors = SelectorTable::build(reachability);
    ProgramTable program_table;

    emit_program_header(
      program, reachability, program_table, selectors, writer, *entry);
    emit_functions(
      context, analysis, reachability, program_table, selectors, writer);

    writer.finish();

    return code;
  }
}
