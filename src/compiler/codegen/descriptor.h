// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "bytecode/generator.h"
#include "compiler/reachability/item.h"
#include "compiler/reachability/reachability.h"
#include "compiler/reachability/selector.h"

namespace verona::compiler
{
  using bytecode::Descriptor;
  using bytecode::Generator;
  using bytecode::Label;

  /**
   * The program table maintains a mapping from items being codegen'ed to their
   * location in the bytecode. Since that location may not be known ahead of
   * time, it uses Generator's relocations to refer to them.
   *
   * The table is populated lazily. The first time an entity or method is looked
   * up, a label is assigned to it.
   */
  struct ProgramTable
  {
    Descriptor find(Generator& gen, const CodegenItem<Entity>& entity)
    {
      auto [it, inserted] = descriptors.insert({entity, Descriptor()});
      if (inserted)
      {
        it->second = gen.create_descriptor();
      }
      return it->second;
    }

    Label
    find(Generator& gen, const CodegenItem<Method>& entity, size_t index = 0)
    {
      auto [it, inserted] = methods.insert({{entity, index}, Label()});
      if (inserted)
      {
        it->second = gen.create_label();
      }
      return it->second;
    }

  private:
    std::map<CodegenItem<Entity>, Descriptor> descriptors;
    std::map<std::pair<CodegenItem<Method>, size_t>, Label> methods;
  };

  void emit_program_header(
    const Program& program,
    const Reachability& reachability,
    ProgramTable& program_table,
    const SelectorTable& selectors,
    Generator& gen,
    const CodegenItem<Method>& main);
};
