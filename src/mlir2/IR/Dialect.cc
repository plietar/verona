#include "mlir2/IR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir2/IR/OpSyntax.h"

#include "llvm/ADT/TypeSwitch.h"

using Type = mlir::Type;

#define GET_TYPEDEF_CLASSES
#include "mlir2/IR/TypeDefs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir2/IR/Ops.cpp.inc"

using namespace mlir;

namespace verona::vam
{
  void VeronaAbstractMachine::initialize()
  {
    addOperations<
#define GET_OP_LIST
#include "mlir2/IR/Ops.cpp.inc"
      >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "mlir2/IR/TypeDefs.cpp.inc"
      >();
    addAttributes<SelectorAttr>();
  }

  Type VeronaAbstractMachine::parseType(::DialectAsmParser& parser) const
  {
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return Type();
    if (Type type = generatedTypeParser(getContext(), parser, keyword))
      return type;

    parser.emitError(parser.getNameLoc(), "invalid 'vam' type: `")
      << keyword << "'";
    return Type();
  }

  void
  VeronaAbstractMachine::printType(Type type, DialectAsmPrinter& printer) const
  {
    if (failed(generatedTypePrinter(type, printer)))
      llvm_unreachable("unknown 'vam' type");
  }

  mlir::Attribute VeronaAbstractMachine::parseAttribute(
    mlir::DialectAsmParser& parser, mlir::Type type) const
  {
    if (type)
    {
      parser.emitError(parser.getNameLoc(), "unexpected type");
      return nullptr;
    }

    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return nullptr;

    if (keyword == "selector")
      return SelectorAttr::parse(parser);

    parser.emitError(parser.getNameLoc(), "invalid 'vam' attribute: `")
      << keyword << "'";
    return nullptr;
  }

  void VeronaAbstractMachine::printAttribute(
    mlir::Attribute attr, mlir::DialectAsmPrinter& os) const
  {
    if (auto selector = attr.dyn_cast<SelectorAttr>())
      selector.print(os);
    else
      llvm_unreachable("unknown 'vam' attribute");
  }

  void DescriptorOp::build(
    mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name)
  {
    state.addAttribute(
      mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    ensureTerminator(*state.addRegion(), builder, state.location);
  }

  DescriptorOp
  DescriptorOp::create(::mlir::Location loc, ::llvm::StringRef name)
  {
    mlir::OpBuilder builder(loc->getContext());
    return builder.create<DescriptorOp>(loc, name);
  }

  MethodOp MethodOp::create(
    ::mlir::Location loc,
    bytecode::SelectorIdx selector,
    mlir::FlatSymbolRefAttr function)
  {
    mlir::OpBuilder builder(loc->getContext());
    return builder.create<MethodOp>(loc, selector, function.getValue());
  }

  static LogicalResult verifyDescriptorSymbolUse(
    mlir::SymbolTableCollection& table,
    mlir::Operation* op,
    mlir::FlatSymbolRefAttr symbol)
  {
    DescriptorOp descriptor =
      table.lookupNearestSymbolFrom<DescriptorOp>(op, symbol);
    if (!descriptor)
      return op->emitOpError("'")
        << symbol.getValue() << "' does not reference a valid descriptor";
    return mlir::success();
  }

  LogicalResult
  NewObjectOp::verifySymbolUses(mlir::SymbolTableCollection& table)
  {
    return verifyDescriptorSymbolUse(table, *this, descriptorAttr());
  }

  LogicalResult
  NewRegionOp::verifySymbolUses(mlir::SymbolTableCollection& table)
  {
    return verifyDescriptorSymbolUse(table, *this, descriptorAttr());
  }

  LogicalResult NewCownOp::verifySymbolUses(mlir::SymbolTableCollection& table)
  {
    return verifyDescriptorSymbolUse(table, *this, descriptorAttr());
  }

  LogicalResult
  LoadDescriptorOp::verifySymbolUses(mlir::SymbolTableCollection& table)
  {
    return verifyDescriptorSymbolUse(table, *this, descriptorAttr());
  }

  LogicalResult DescriptorOp::verifyOp()
  {
    llvm::DenseMap<mlir::Attribute, MethodOp> methods;
    llvm::DenseMap<mlir::Attribute, FieldOp> fields;

    for (auto& op : *this)
    {
      auto result = llvm::TypeSwitch<Operation*, LogicalResult>(&op)
                      .Case<MethodOp>([&](MethodOp method) {
                        auto [it, inserted] =
                          methods.try_emplace(method.selectorAttr(), method);
                        if (!inserted)
                        {
                          method->emitOpError("Selector ")
                            << method.selectorAttr() << " is already used";
                          return mlir::failure();
                        }
                        return mlir::success();
                      })
                      .Case<FieldOp>([&](FieldOp field) {
                        auto [it, inserted] =
                          fields.try_emplace(field.selectorAttr(), field);
                        if (!inserted)
                        {
                          field->emitOpError("Selector ")
                            << field.selectorAttr() << " is already used";
                          return mlir::failure();
                        }
                        return mlir::success();
                      })
                      .Default([](Operation*) { return mlir::success(); });

      if (mlir::failed(result))
        return mlir::failure();
    }
    return mlir::success();
  }
}
