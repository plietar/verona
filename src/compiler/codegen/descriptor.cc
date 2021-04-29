// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/descriptor.h"

#include "ds/helpers.h"

namespace verona::compiler
{
  using bytecode::SelectorIdx;

  class EmitProgramHeader
  {
  public:
    EmitProgramHeader(
      const Program& program,
      const Reachability& reachability,
      ProgramTable& program_table,
      const SelectorTable& selectors,
      Generator& gen)
    : program(program),
      reachability(reachability),
      program_table(program_table),
      selectors(selectors),
      gen(gen)
    {}

    void emit_descriptor_table()
    {
      // Number of descriptors
      gen.u32(truncate<uint32_t>(reachability.entities.size()));

      size_t index = 0;
      for (const auto& [entity, info] : reachability.entities)
      {
        Descriptor descriptor = program_table.find(gen, entity);
        gen.define_relocatable(descriptor, index++);
        emit_descriptor(entity, info);
      }
    }

    void emit_descriptor(
      const CodegenItem<Entity>& entity, const EntityReachability& info)
    {
      switch (entity.definition->kind->value())
      {
        case Entity::Class:
        case Entity::Primitive:
          emit_class_primitive_descriptor(entity, info);
          break;

        case Entity::Interface:
          emit_interface_descriptor(entity, info);
          break;

          EXHAUSTIVE_SWITCH;
      }
    }

    void emit_class_primitive_descriptor(
      const CodegenItem<Entity>& entity, const EntityReachability& info)
    {
      Generator::Relocatable rel_fields = gen.create_relocatable();

      gen.str(entity.instantiated_path());
      gen.u32(truncate<uint32_t>(info.methods.size()));
      gen.u32(rel_fields);
      gen.u32(truncate<uint32_t>(info.subtypes.size()));

      // Output label for finaliser for this class, if it has one.
      if (info.finaliser.has_value())
      {
        Label label = program_table.find(gen, *info.finaliser);
        gen.u32(label);
      }
      else
      {
         gen.u32(0);
      }

      emit_methods(info);
      uint32_t field_count = emit_fields(entity);
      emit_subtypes(info);

      gen.define_relocatable(rel_fields, field_count);
    }

    void emit_interface_descriptor(
      const CodegenItem<Entity>& entity, const EntityReachability& info)
    {
      gen.str(entity.instantiated_path());
      gen.u32(0); // method_slots
      gen.u32(0); // method_count
      gen.u32(0); // field_slots
      gen.u32(0); // field_count
      gen.u32(truncate<uint32_t>(info.subtypes.size()));
      gen.u32(0); // finaliser
      emit_subtypes(info);
    }

    /// For each field in the class, emit it's selector index. This is used by
    /// the VM to construct the field VTable.
    ///
    /// Returns the number of fields.
    uint32_t emit_fields(const CodegenItem<Entity>& entity)
    {
      uint32_t field_count = 0;

      for (const auto& member : entity.definition->members)
      {
        if (const Field* fld = member->get_as<Field>())
        {
          SelectorIdx index = selectors.get(Selector::field(fld->name));
          gen.selector(index);
          field_count++;
        }
      }

      return field_count;
    }

    /// For each instantiation of a method in the class, emit it's selector
    /// index and offset into the program. This is used by the VM to construct
    /// the field VTable.
    void emit_methods(const EntityReachability& info)
    {
      for (const auto& method : info.methods)
      {
        TypeList arguments;
        for (const auto& param : method.definition->signature->generics->types)
        {
          arguments.push_back(method.instantiation.types().at(param->index));
        }

        Label label = program_table.find(gen, method);
        Selector selector =
          Selector::method(method.definition->name, arguments);
        SelectorIdx index = selectors.get(selector);
        gen.selector(index);
        gen.u32(label);
      }
    }

    /// For each subtype of the entity, emit the corresponding descriptor index.
    void emit_subtypes(const EntityReachability& info)
    {
      for (const auto& subtype : info.subtypes)
      {
        const auto& subtype_info = reachability.entities.at(subtype);
        Descriptor descriptor = program_table.find(gen, subtype);
        gen.u32(descriptor);
      }
    }

    void emit_optional_special_descriptor(const std::string& name)
    {
      const EntityReachability* entity_info = nullptr;
      if (const Entity* entity = program.find_entity(name))
      {
        CodegenItem item(entity, Instantiation::empty());
        if (reachability.is_reachable(item))
        {
          Descriptor descriptor = program_table.find(gen, item);
          gen.u32(descriptor);
          return;
        }
      }

      gen.u32(bytecode::DescriptorIdx::invalid().value);
    }

    void emit_special_descriptors(const CodegenItem<Entity>& main_class)
    {
      // Index of the main descriptor
      gen.u32(program_table.find(gen, main_class));

      // Index of the main selector
      gen.selector(selectors.get(Selector::method("main", TypeList())));

      emit_optional_special_descriptor("U64");
      emit_optional_special_descriptor("String");
    }

  private:
    const Program& program;
    const Reachability& reachability;
    ProgramTable& program_table;
    const SelectorTable& selectors;
    Generator& gen;
  };

  void emit_program_header(
    const Program& program,
    const Reachability& reachability,
    ProgramTable& program_table,
    const SelectorTable& selectors,
    Generator& gen,
    const CodegenItem<Entity>& main)
  {
    EmitProgramHeader emit(
      program, reachability, program_table, selectors, gen);
    emit.emit_descriptor_table();
    emit.emit_special_descriptors(main);
  }
};
