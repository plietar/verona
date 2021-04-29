// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/codegen.h"

#include "bytecode/bytecode.h"
#include "bytecode/writer.h"
#include "compiler/ast.h"
#include "compiler/codegen/builtins.h"
#include "compiler/codegen/descriptor.h"
#include "compiler/codegen/ir.h"
#include "compiler/reachability/reachability.h"

namespace verona::compiler
{
  using bytecode::BytecodeWriter;

  /**
   * Writes the magic numbers to the bytecode
   * @param code BytecodeWriter to which the bytes should be emitted
   */
  void write_magic_number(BytecodeWriter& code)
  {
    code.u32(bytecode::MAGIC_NUMBER);
  }

  void emit_functions(
    Context& context,
    const AnalysisResults& analysis,
    const Reachability& reachability,
    ProgramTable& program_table,
    const SelectorTable& selectors,
    BytecodeWriter& writer)
  {
    for (const auto& [entity, entity_info] : reachability.entities)
    {
      for (const auto& method : entity_info.methods)
      {
        if (method.definition->kind() == Method::Builtin)
        {
          generate_builtin(writer, program_table, method);
        }
        else
        {
          const FnAnalysis& fn_analysis =
            analysis.functions.at(method.definition);
          generate_ir_function(
            context, program_table, selectors, writer, method, fn_analysis);
        }
      }
    }
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
