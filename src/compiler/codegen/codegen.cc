// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/codegen.h"

#include "compiler/ast.h"
#include "compiler/codegen/descriptor.h"
#include "compiler/codegen/function.h"
#include "compiler/codegen/generator.h"
#include "compiler/codegen/reachability.h"
#include "interpreter/bytecode.h"

namespace verona::compiler
{
  /**
   * Writes the magic numbers to the bytecode
   * @param code Generator to which the bytes should be emitted
   */
  void write_magic_number(Generator& code)
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

    Generator gen(code);
    write_magic_number(gen);

    Reachability reachability =
      compute_reachability(context, program, *entry, analysis);
    SelectorTable selectors = SelectorTable::build(reachability);
    ProgramTable program_table;

    emit_program_header(
      program, reachability, program_table, selectors, gen, *entry);
    emit_functions(
      context, analysis, reachability, program_table, selectors, gen);

    gen.finish();

    return code;
  }
}
