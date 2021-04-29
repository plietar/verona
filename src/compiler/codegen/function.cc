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
  using bytecode::DescriptorIdx;
  using bytecode::Opcode;

  RegisterAllocator::RegisterAllocator(const FunctionABI& abi)
  : next_register_(abi.callspace())
  {}

  Register RegisterAllocator::get()
  {
    if (next_register_ + children_call_space_ >= bytecode::REGISTER_COUNT)
      throw std::logic_error("Ran out of registers");

    return Register(truncate<uint8_t>(next_register_++));
  }

  void RegisterAllocator::reserve_child_callspace(const FunctionABI& abi)
  {
    if (next_register_ + abi.callspace() >= bytecode::REGISTER_COUNT)
      throw std::logic_error("Ran out of registers");

    children_call_space_ = std::max(children_call_space_, abi.callspace());
  }

  uint8_t RegisterAllocator::frame_size() const
  {
    return truncate<uint8_t>(next_register_ + children_call_space_);
  }

  void emit_functions(
    Context& context,
    const AnalysisResults& analysis,
    const Reachability& reachability,
    ProgramTable& program_table,
    const SelectorTable& selectors,
    Generator& gen)
  {
    for (const auto& [entity, entity_info] : reachability.entities)
    {
      for (const auto& method : entity_info.methods)
      {
        if (method.definition->kind() == Method::Builtin)
        {
          generate_builtin(gen, program_table, method);
        }
        else
        {
          const FnAnalysis& fn_analysis =
            analysis.functions.at(method.definition);
          generate_ir_function(
            context, program_table, selectors, gen, method, fn_analysis);
        }
      }
    }
  }
}
