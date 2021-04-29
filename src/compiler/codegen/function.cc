// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/function.h"

#include "compiler/codegen/builtins.h"
#include "compiler/codegen/ir.h"
#include "compiler/printing.h"
#include "compiler/typecheck/typecheck.h"
#include "compiler/visitor.h"
#include "compiler/zip.h"

namespace verona::compiler
{
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
}
