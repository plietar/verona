// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "bytecode/writer.h"
#include "compiler/reachability/item.h"
#include "compiler/reachability/reachability.h"
#include "compiler/reachability/selector.h"

namespace verona::compiler
{
  /**
   * The program table maintains a mapping from items being codegen'ed to their
   * location in the bytecode. Since that location may not be known ahead of
   * time, it uses BytecodeWriter's relocations to refer to them.
   *
   * The table is populated lazily. The first time an entity or method is looked
   * up, a label is assigned to it.
   */
  struct ProgramTable
  {
    bytecode::Descriptor
    find(bytecode::BytecodeWriter& writer, const CodegenItem<Entity>& entity)
    {
      auto [it, inserted] =
        descriptors.insert({entity, bytecode::Descriptor()});
      if (inserted)
      {
        it->second = writer.create_descriptor();
      }
      return it->second;
    }

    bytecode::Label find(
      bytecode::BytecodeWriter& writer,
      const CodegenItem<Method>& entity,
      size_t index = 0)
    {
      auto [it, inserted] =
        methods.insert({{entity, index}, bytecode::Label()});
      if (inserted)
      {
        it->second = writer.create_label();
      }
      return it->second;
    }

  private:
    std::map<CodegenItem<Entity>, bytecode::Descriptor> descriptors;
    std::map<std::pair<CodegenItem<Method>, size_t>, bytecode::Label> methods;
  };

  void emit_program_header(
    const Program& program,
    const Reachability& reachability,
    ProgramTable& program_table,
    const SelectorTable& selectors,
    bytecode::BytecodeWriter& writer,
    const CodegenItem<Method>& main);
};
