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
      BytecodeWriter& writer)
    : program(program),
      reachability(reachability),
      program_table(program_table),
      selectors(selectors),
      writer(writer)
    {}

    void emit_descriptor_table()
    {
      // Number of descriptors
      writer.u32(truncate<uint32_t>(reachability.entities.size()));

      size_t index = 0;
      for (const auto& [entity, info] : reachability.entities)
      {
        Descriptor descriptor = program_table.find(writer, entity);
        writer.define_relocatable(descriptor, index++);
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
      BytecodeWriter::Relocatable rel_fields = writer.create_relocatable();

      writer.str(entity.instantiated_path());
      writer.u32(truncate<uint32_t>(info.methods.size()));
      writer.u32(rel_fields);
      writer.u32(truncate<uint32_t>(info.subtypes.size()));

      // Output label for finaliser for this class, if it has one.
      if (info.finaliser.has_value())
      {
        Label label = program_table.find(writer, *info.finaliser);
        writer.u32(label);
      }
      else
      {
        writer.u32(0);
      }

      emit_methods(info);
      uint32_t field_count = emit_fields(entity);
      emit_subtypes(info);

      writer.define_relocatable(rel_fields, field_count);
    }

    void emit_interface_descriptor(
      const CodegenItem<Entity>& entity, const EntityReachability& info)
    {
      writer.str(entity.instantiated_path());
      writer.u32(0); // method_slots
      writer.u32(0); // method_count
      writer.u32(0); // field_slots
      writer.u32(0); // field_count
      writer.u32(truncate<uint32_t>(info.subtypes.size()));
      writer.u32(0); // finaliser
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
          writer.selector(index);
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
        bytecode::SelectorIdx index = selectors.get(method.selector());
        Label label = program_table.find(writer, method);
        writer.selector(index);
        writer.u32(label);
      }
    }

    /// For each subtype of the entity, emit the corresponding descriptor index.
    void emit_subtypes(const EntityReachability& info)
    {
      for (const auto& subtype : info.subtypes)
      {
        const auto& subtype_info = reachability.entities.at(subtype);
        Descriptor descriptor = program_table.find(writer, subtype);
        writer.u32(descriptor);
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
          Descriptor descriptor = program_table.find(writer, item);
          writer.u32(descriptor);
          return;
        }
      }

      writer.u32(bytecode::DescriptorIdx::invalid().value);
    }

    void emit_special_descriptors(const CodegenItem<Method>& main)
    {
      // Index of the Main class descriptor
      writer.u32(program_table.find(writer, main.parent()));

      // Index of the main selector
      writer.selector(selectors.get(main.selector()));

      emit_optional_special_descriptor("U64");
      emit_optional_special_descriptor("String");
    }

  private:
    const Program& program;
    const Reachability& reachability;
    ProgramTable& program_table;
    const SelectorTable& selectors;
    BytecodeWriter& writer;
  };

  void emit_program_header(
    const Program& program,
    const Reachability& reachability,
    ProgramTable& program_table,
    const SelectorTable& selectors,
    BytecodeWriter& writer,
    const CodegenItem<Method>& main)
  {
    EmitProgramHeader emit(
      program, reachability, program_table, selectors, writer);
    emit.emit_descriptor_table();
    emit.emit_special_descriptors(main);
  }
};
