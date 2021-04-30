#include "compiler/codegen/codegen.h"
#include "compiler/ir/ir.h"
#include "compiler/reachability/reachability.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

namespace verona::vam
{
  using namespace verona::compiler;

  struct BlockMap
  {
    BlockMap(mlir::Region& region) : region(region) {}

    mlir::Block* operator()(const BasicBlock* bb)
    {
      auto [it, inserted] = mapping.insert({bb, nullptr});
      if (inserted)
      {
        it->second = new mlir::Block;
        region.push_back(it->second);
      }
      return it->second;
    }

  private:
    std::unordered_map<const BasicBlock*, mlir::Block*> mapping;
    mlir::Region& region;
  };

  struct VariableMap
  {
    void define(Variable variable, mlir::Value value)
    {
      mapping.insert({variable, value});
    }

    mlir::Value operator()(Variable variable)
    {
      return mapping.at(variable);
    }

    template<typename V>
    std::vector<mlir::Value> operator()(const std::vector<V>& variables)
    {
      std::vector<mlir::Value> result;
      for (auto v : variables)
      {
        result.push_back(operator()(v));
      }
      return result;
    }

    std::unordered_map<Variable, mlir::Value> mapping;
  };

  class Lowering
  {
    struct Visitor
    {
      Visitor(
        Lowering* parent,
        mlir::ImplicitLocOpBuilder& builder,
        BlockMap& blocks,
        VariableMap& variables,
        const TypecheckResults& typecheck,
        const Instantiation& instantiation)
      : parent(parent),
        builder(builder),
        blocks(blocks),
        variables(variables),
        typecheck(typecheck),
        instantiation(instantiation)
      {}

      TypeList reify(const TypeList& arguments)
      {
        return instantiation.apply(parent->context, arguments);
      }

      TypePtr reify(const TypePtr& type)
      {
        return instantiation.apply(parent->context, type);
      }

      TypeList reify(TypeArgumentsId id)
      {
        return reify(typecheck.type_arguments.at(id));
      }

      void operator()(const MatchTerminator& term)
      {
        abort();
      }
      void operator()(const WhenStmt& stmt)
      {
        abort();
      }
      void operator()(const MatchBindStmt& stmt)
      {
        abort();
      }

      void operator()(const ReturnTerminator& term)
      {
        builder.create<ReturnOp>(variables(term.input));
      }

      void operator()(const BranchTerminator& term)
      {
        builder.create<mlir::BranchOp>(blocks(term.target));
      }

      void operator()(const IfTerminator& term)
      {
        auto t = builder.create<TruthinessOp>(
          builder.getI1Type(), variables(term.input));
        builder.create<mlir::CondBranchOp>(
          t, blocks(term.true_target), blocks(term.false_target));
      }

      void operator()(const IntegerLiteralStmt& stmt)
      {
        auto value = builder.create<LiteralOp>(
          valueType(), builder.getI64IntegerAttr(stmt.value));
        variables.define(stmt.output, value);
      }

      void operator()(const StringLiteralStmt& stmt)
      {
        auto value = builder.create<LiteralOp>(
          valueType(), builder.getStringAttr(stmt.value));
        variables.define(stmt.output, value);
      }

      void operator()(const UnitStmt& stmt)
      {
        auto value =
          builder.create<LiteralOp>(valueType(), builder.getUnitAttr());
        variables.define(stmt.output, value);
      }

      void operator()(const EndScopeStmt& stmt)
      {
        for (auto v : stmt.dead_variables)
        {
          builder.create<DropOp>(variables(v));
        }
      }

      void operator()(const OverwriteStmt& stmt)
      {
        builder.create<DropOp>(variables(stmt.dead_variable));
      }

      void operator()(const CopyStmt& stmt)
      {
        auto value = builder.create<CopyOp>(valueType(), variables(stmt.input));
        variables.define(stmt.output, value);
      }

      void operator()(const ViewStmt& stmt)
      {
        auto value =
          builder.create<MutViewOp>(valueType(), variables(stmt.input));
        variables.define(stmt.output, value);
      }

      void operator()(const NewStmt& stmt)
      {
        CodegenItem<Entity> entity(
          stmt.definition, Instantiation(reify(stmt.type_arguments)));
        mlir::FlatSymbolRefAttr symbol = mlir::FlatSymbolRefAttr::get(
          entity.instantiated_path(), parent->mlir_context);

        mlir::Value value;
        if (stmt.parent.has_value())
        {
          value = builder.create<NewObjectOp>(
            valueType(), symbol, variables(*stmt.parent));
        }
        else
        {
          value = builder.create<NewRegionOp>(valueType(), symbol);
        }
        variables.define(stmt.output, value);
      }

      void operator()(const StaticTypeStmt& stmt)
      {
        CodegenItem<Entity> entity(
          stmt.definition, Instantiation(reify(stmt.type_arguments)));
        mlir::FlatSymbolRefAttr symbol = mlir::FlatSymbolRefAttr::get(
          entity.instantiated_path(), parent->mlir_context);
        mlir::Value value =
          builder.create<LoadDescriptorOp>(valueType(), symbol);
        variables.define(stmt.output, value);
      }

      void operator()(const ReadFieldStmt& stmt)
      {
        bytecode::SelectorIdx selector =
          parent->selectors.get(Selector::field(stmt.name));
        mlir::Value value = builder.create<LoadFieldOp>(
          valueType(), variables(stmt.base), selector);
        variables.define(stmt.output, value);
      }

      void operator()(const WriteFieldStmt& stmt)
      {
        bytecode::SelectorIdx selector =
          parent->selectors.get(Selector::field(stmt.name));
        mlir::Value value = builder.create<StoreFieldOp>(
          valueType(), variables(stmt.base), selector, variables(stmt.right));
        variables.define(stmt.output, value);
      }

      void operator()(const CallStmt& stmt)
      {
        bytecode::SelectorIdx selector = parent->selectors.get(
          Selector::method(stmt.method, reify(stmt.type_arguments)));
        auto value = builder.create<CallOp>(
          valueType(),
          variables(stmt.receiver),
          selector,
          variables(stmt.arguments));
        variables.define(stmt.output, value);
      }

      mlir::Type valueType()
      {
        return parent->valueType;
      }

      Lowering* parent;
      mlir::ImplicitLocOpBuilder& builder;
      BlockMap& blocks;
      VariableMap& variables;
      const TypecheckResults& typecheck;
      const Instantiation& instantiation;
    };

  public:
    Lowering(
      mlir::MLIRContext* mlir_context,
      Context& context,
      const AnalysisResults& analysis,
      const SelectorTable& selectors)
    : mlir_context(mlir_context),
      context(context),
      analysis(analysis),
      selectors(selectors),
      valueType(ValueType::get(mlir_context))
    {}

    mlir::Type valueType;

    mlir::Location getUnknownLoc()
    {
      return mlir::UnknownLoc::get(mlir_context);
    }

    void lower(
      mlir::Region& body,
      const TypecheckResults& typecheck,
      const Instantiation& instantiation,
      FunctionIR& ir)
    {
      BlockMap blocks(body);
      VariableMap variables;
      IRTraversal traversal(ir);

      mlir::Block* entry = blocks(ir.entry);
      auto receiver = entry->addArgument(valueType);
      if (ir.receiver.has_value())
      {
        variables.define(*ir.receiver, receiver);
      }
      for (Variable param : ir.parameters)
      {
        variables.define(param, entry->addArgument(valueType));
      }

      while (BasicBlock* bb = traversal.next())
      {
        mlir::Block* block = blocks(bb);
        if (bb != ir.entry)
        {
          // Define block parameters
        }

        auto builder =
          mlir::ImplicitLocOpBuilder::atBlockBegin(getUnknownLoc(), block);
        Visitor visitor(
          this, builder, blocks, variables, typecheck, instantiation);

        for (const auto& stmt : bb->statements)
        {
          std::visit(visitor, stmt);
        }
        std::visit(visitor, *bb->terminator);
      }
    }

    void lower_builtin(mlir::Region& body, std::string_view name)
    {
      mlir::Block* block = new mlir::Block;
      body.push_back(block);
      auto builder =
        mlir::ImplicitLocOpBuilder::atBlockBegin(getUnknownLoc(), block);
      if (name == "Builtin.print")
      {
        block->addArgument(valueType);
        auto msg = block->addArgument(valueType);

        builder.create<PrintOp>(msg, mlir::ValueRange());
        builder.create<DropOp>(msg);
        auto unit = builder.create<LiteralOp>(valueType, builder.getUnitAttr());
        builder.create<ReturnOp>(unit);
      }
    }

    MethodOp lower(const CodegenItem<Method>& method)
    {
      bytecode::SelectorIdx selector = selectors.get(method.selector());
      MethodOp method_op =
        MethodOp::create(getUnknownLoc(), selector, method.instantiated_path());

      if (method.definition->kind() == Method::Builtin)
      {
        lower_builtin(method_op.body(), method.instantiated_path());
      }
      else
      {
        const FnAnalysis& fn_analysis =
          analysis.functions.at(method.definition);
        MethodIR& mir = *fn_analysis.ir;
        FunctionIR& ir = *mir.function_irs.at(0);
        lower(
          method_op.body(), *fn_analysis.typecheck, method.instantiation, ir);
      }

      return method_op;
    }

    DescriptorOp lower(CodegenItem<Entity> entity, EntityReachability info)
    {
      DescriptorOp op =
        DescriptorOp::create(getUnknownLoc(), entity.instantiated_path());
      for (auto method : info.methods)
      {
        op.push_back(lower(method));
      }
      return op;
    }

  private:
    mlir::MLIRContext* mlir_context;
    Context& context;

    const AnalysisResults& analysis;
    const SelectorTable& selectors;
  };

  mlir::OwningModuleRef lower(
    mlir::MLIRContext* mlir_context,
    Context& context,
    const Program& program,
    const AnalysisResults& analysis)
  {
    auto entry = find_program_entry(context, program);
    if (!entry)
      return nullptr;

    auto reachability =
      compute_reachability(context, program, *entry, analysis);
    auto selectors = SelectorTable::build(reachability);

    mlir::OpBuilder builder(mlir_context);
    mlir::OwningModuleRef module =
      mlir::ModuleOp::create(builder.getUnknownLoc());

    bytecode::SelectorIdx main_selector = selectors.get(entry->selector());
    (*module)->setAttr(
      "vam.main_class",
      mlir::FlatSymbolRefAttr::get(
        entry->parent().instantiated_path(), mlir_context));
    (*module)->setAttr(
      "vam.main_method", SelectorAttr::get(main_selector, mlir_context));

    Lowering lowering(mlir_context, context, analysis, selectors);
    for (auto [entity, info] : reachability.entities)
    {
      module->push_back(lowering.lower(entity, info));
    }

    return module;
  }
}
