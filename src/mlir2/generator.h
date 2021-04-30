#include "bytecode/generator.h"
#include "bytecode/writer.h"
#include "ds/error.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/TypeSwitch.h"

#include <unordered_map>

namespace verona::vam
{
  using bytecode::Label;
  using bytecode::Opcode;

  struct ProgramTable
  {
    llvm::DenseMap<llvm::StringRef, bytecode::DescriptorIdx> descriptors;
    std::vector<std::pair<MethodOp, Label>> methods;
  };

  static bytecode::DescriptorIdx find_descriptor(
    const ProgramTable& table, mlir::ModuleOp module, std::string_view key)
  {
    auto symbol = module->getAttrOfType<mlir::FlatSymbolRefAttr>(key);
    if (symbol)
    {
      auto it = table.descriptors.find(symbol.getValue());
      if (it != table.descriptors.end())
      {
        return it->second;
      }
    }

    return bytecode::DescriptorIdx::invalid();
  }

  static void emit_header(
    mlir::ModuleOp module,
    bytecode::BytecodeWriter& writer,
    ProgramTable& table)
  {
    auto entities_rel = writer.create_relocatable();
    writer.u32(bytecode::MAGIC_NUMBER);
    writer.u32(entities_rel); // entity count
    uint32_t entities = 0;
    for (DescriptorOp descriptor : module.getOps<DescriptorOp>())
    {
      table.descriptors.insert(
        {descriptor.getName(), bytecode::DescriptorIdx(entities)});
      auto methods_rel = writer.create_relocatable();
      auto fields_rel = writer.create_relocatable();

      writer.str(descriptor.sym_name());
      writer.u32(methods_rel);
      writer.u32(fields_rel);
      writer.u32(0); // Number of subtypes

      std::optional<std::pair<bytecode::SelectorIdx, Label>> finaliser;
      if (descriptor.finaliser().hasValue())
      {
        Label label = writer.create_label();
        writer.u32(label);
        finaliser = {*descriptor.finaliser(), label};
      }
      else
      {
        writer.u32(0);
      }

      uint32_t methods = 0;
      uint32_t fields = 0;
      for (MethodOp method : descriptor.getOps<MethodOp>())
      {
        Label entry;

        if (finaliser && method.selector() == finaliser->first)
        {
          entry = finaliser->second;
          finaliser.reset();
        }
        else
        {
          entry = writer.create_label();
        }

        writer.selector(method.selector());
        writer.u32(entry); // offset
        table.methods.push_back({method, entry});
        methods += 1;
      }

      if (finaliser.has_value())
      {
        InternalError::print(
          "Did not find a finaliser for descriptor '{}' with selector '{}'\n",
          descriptor.getName(),
          finaliser->first);
      }

      for (FieldOp field : descriptor.getOps<FieldOp>())
      {
        writer.selector(field.selector()); // field index
        fields += 1;
      }
      writer.define_relocatable(methods_rel, methods);
      writer.define_relocatable(fields_rel, fields);
      entities += 1;
    }

    auto main_class = find_descriptor(table, module, "vam.main_class");
    auto main_method = module->getAttrOfType<SelectorAttr>("vam.main_method");
    if (!main_class.is_valid() || !main_method)
    {
      InternalError::print("Missing main class or method attribute");
    }

    writer.u32(main_class.value);
    writer.selector(main_method.getValue());
    writer.u32(find_descriptor(table, module, "vam.u64_class").value);
    writer.u32(find_descriptor(table, module, "vam.string_class").value);

    writer.define_relocatable(entities_rel, entities);
  }

  using WorkQueue = std::deque<std::tuple<Label, mlir::Region*, std::string>>;

  struct Generator
  : public bytecode::
      FunctionGenerator<mlir::Value, mlir::Block*, llvm::DenseMap>
  {
    Generator(
      bytecode::BytecodeWriter& writer,
      const ProgramTable& table,
      WorkQueue& queue,
      std::string_view name,
      bytecode::FunctionABI abi)
    : bytecode::FunctionGenerator<mlir::Value, mlir::Block*, llvm::DenseMap>(
        writer, name, abi),
      table(table),
      queue(queue)
    {}

    const ProgramTable& table;
    WorkQueue& queue;

    void load_descriptor(bytecode::Register result, llvm::StringRef name)
    {
      auto it = table.descriptors.find(name);
      if (it == table.descriptors.end())
        InternalError::print("Cannot find descriptor for symbol '{}'\n", name);
      emit<Opcode::LoadDescriptor>(result, it->second);
    }

    bytecode::Register load_descriptor(llvm::StringRef name)
    {
      bytecode::Register descriptor = allocator().get();
      load_descriptor(descriptor, name);
      return descriptor;
    }

    void generate(LiteralOp op)
    {
      bytecode::Register result = variable(op.result());
      llvm::TypeSwitch<mlir::Attribute>(op.value())
        .Case<mlir::StringAttr>([&](mlir::StringAttr attr) {
          emit<Opcode::String>(result, attr.getValue());
        })
        .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr attr) {
          emit<Opcode::Int64>(result, attr.getValue().getZExtValue());
        })
        .Case<mlir::UnitAttr>(
          [&](mlir::UnitAttr attr) { emit<Opcode::Clear>(result); })
        .Default(
          [&](mlir::Attribute attr) { llvm_unreachable("unknown attribute"); });
    }

    void generate(PrintOp op)
    {
      emit<Opcode::Print>(variable(op.format()), variables(op.values()));
    }

    void generate(ReturnOp op)
    {
      emit<Opcode::Move>(bytecode::Register(0), variable(op.operand()));
      emit<Opcode::Return>();
    }

    void generate(DropOp op)
    {
      emit<Opcode::Clear>(variable(op.operand()));
    }

    void generate(CopyOp op)
    {
      emit<Opcode::Copy>(variable(op.result()), variable(op.operand()));
    }

    void generate(MutViewOp op)
    {
      emit<Opcode::MutView>(variable(op.result()), variable(op.operand()));
    }

    void generate(CallOp op)
    {
      bytecode::FunctionABI child_abi(1 + op.args().size(), 1);
      allocator().reserve_child_callspace(child_abi);

      size_t index = 0;

      emit<Opcode::Copy>(
        callee_register(child_abi, truncate<uint8_t>(index++)),
        variable(op.receiver()));
      for (const auto& var : op.args())
      {
        bytecode::Register src = variable(var);
        bytecode::CalleeRegister dst =
          callee_register(child_abi, truncate<uint8_t>(index++));
        emit<Opcode::Copy>(dst, src);
      }

      emit<Opcode::Call>(
        op.selector(), truncate<uint8_t>(child_abi.callspace()));
      emit<Opcode::Copy>(
        variable(op.result()),
        callee_register(child_abi, truncate<uint8_t>(0)));
    }

    void generate(NewRegionOp op)
    {
      bytecode::Register result = variable(op.result());
      bytecode::Register descriptor = load_descriptor(op.descriptor());
      emit<Opcode::NewRegion>(result, descriptor);
    }

    void generate(NewObjectOp op)
    {
      bytecode::Register result = variable(op.result());
      bytecode::Register descriptor = load_descriptor(op.descriptor());
      emit<Opcode::NewObject>(result, variable(op.parent()), descriptor);
    }

    void generate(TraceRegionOp op)
    {
      emit<Opcode::TraceRegion>(variable(op.operand()));
    }

    void generate(LoadFieldOp op)
    {
      emit<Opcode::Load>(
        variable(op.result()), variable(op.origin()), op.selector());
    }

    void generate(StoreFieldOp op)
    {
      emit<Opcode::Store>(
        variable(op.result()),
        variable(op.origin()),
        op.selector(),
        variable(op.value()));
    }

    void generate(ProtectOp op)
    {
      emit<Opcode::Protect>(variables(op.operands()));
    }

    void generate(UnprotectOp op)
    {
      emit<Opcode::Unprotect>(variables(op.operands()));
    }

    void generate(TruthinessOp op)
    {
      emit<Opcode::Copy>(variable(op.result()), variable(op.operand()));
    }

    void generate(mlir::BranchOp op)
    {
      emit<Opcode::Jump>(label(op.dest()));
    }

    void generate(mlir::CondBranchOp op)
    {
      emit<Opcode::JumpIf>(variable(op.condition()), label(op.trueDest()));
      emit<Opcode::Jump>(label(op.falseDest()));
    }

    void generate(WhenOp op)
    {
      bytecode::FunctionABI abi(1, 1);
      allocator().reserve_child_callspace(abi);

      Label label = create_label();
      emit<Opcode::When>(label, uint8_t(0), uint8_t(0));

      queue.emplace_back(label, &op.body(), "behaviour");
    }

    void generate(NewCownOp op)
    {
      bytecode::Register descriptor = load_descriptor(op.descriptor());
      emit<Opcode::NewCown>(
        variable(op.result()), descriptor, variable(op.operand()));
    }

    void generate(LoadDescriptorOp op)
    {
      load_descriptor(variable(op.result()), op.descriptor());
    }

    void generate(mlir::Region& region)
    {
      for (mlir::Block& block : region)
      {
        define_label(&block);

        for (mlir::Operation& op : block)
        {
          llvm::TypeSwitch<mlir::Operation*>(&op)
            .Case<
              LiteralOp,
              PrintOp,
              ReturnOp,
              DropOp,
              CopyOp,
              MutViewOp,
              CallOp,
              NewRegionOp,
              NewObjectOp,
              NewCownOp,
              LoadDescriptorOp,
              TraceRegionOp,
              LoadFieldOp,
              StoreFieldOp,
              ProtectOp,
              UnprotectOp,
              TruthinessOp,
              WhenOp,
              mlir::CondBranchOp,
              mlir::BranchOp>([&](auto innerOp) { generate(innerOp); })
            .Default([](mlir::Operation* op) {
              llvm::errs() << *op;
              llvm_unreachable("unknown operation");
            });
        }
      }
    }
  };

  static void emit_method(
    MethodOp method,
    Label label,
    bytecode::BytecodeWriter& writer,
    ProgramTable& table)
  {
    WorkQueue queue;
    queue.emplace_back(label, &method.body(), method.name());

    while (!queue.empty())
    {
      auto [label, region, name] = queue.front();
      queue.pop_front();

      writer.define_label(label);
      bytecode::FunctionABI abi(region->getArguments().size(), 1);
      Generator gen(writer, table, queue, name, abi);
      gen.bind_parameters(region->getArguments());
      gen.generate(*region);
      gen.finish();
    }
  }

  std::vector<uint8_t> generate(mlir::ModuleOp module)
  {
    std::vector<uint8_t> bytecode;

    bytecode::BytecodeWriter gen(bytecode);
    ProgramTable table;

    emit_header(module, gen, table);
    for (const auto& [op, label] : table.methods)
    {
      emit_method(op, label, gen, table);
    }
    gen.finish();

    return bytecode;
  }
}
