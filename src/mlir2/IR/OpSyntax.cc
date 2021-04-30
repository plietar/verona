#include "mlir2/IR/OpSyntax.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir2/IR/Dialect.h"

using namespace mlir;

namespace verona::vam
{
  static ParseResult parseArgument(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::OperandType>& argNames,
    SmallVectorImpl<Type>& argTypes,
    Type defaultType)
  {
    OpAsmParser::OperandType name;
    Type type;
    if (failed(parser.parseRegionArgument(name)))
    {
      return failure();
    }

    if (succeeded(parser.parseOptionalColon()))
    {
      if (failed(parser.parseType(type)))
      {
        return failure();
      }
    }
    else
    {
      type = defaultType;
    }
    argNames.push_back(name);
    argTypes.push_back(type);
    return success();
  }

  static ParseResult parseArgumentList(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::OperandType>& argNames,
    SmallVectorImpl<Type>& argTypes,
    Type defaultType)
  {
    if (parser.parseLParen())
      return failure();

    if (failed(parser.parseOptionalRParen()))
    {
      do
      {
        if (parseArgument(parser, argNames, argTypes, defaultType))
        {
          return failure();
        }
      } while (succeeded(parser.parseOptionalComma()));
      parser.parseRParen();
    }

    return success();
  }

  static void printArgumentList(
    OpAsmPrinter& printer, ArrayRef<BlockArgument> arguments, Type defaultType)
  {
    printer << "(";
    bool first = true;
    for (const Value& value : arguments)
    {
      if (first)
      {
        first = false;
      }
      else
      {
        printer << ", ";
      }

      printer.printOperand(value);
      if (value.getType() != defaultType)
      {
        printer << ": " << value.getType();
      }
    }
    printer << ")";
  }

  mlir::ParseResult parseVeronaBody(
    mlir::OpAsmParser& parser,
    mlir::Region& region,
    std::optional<std::reference_wrapper<mlir::NamedAttrList>> attrs)
  {
    SmallVector<OpAsmParser::OperandType, 4> argNames;
    SmallVector<Type, 4> argTypes;

    if (failed(parseArgumentList(
          parser,
          argNames,
          argTypes,
          ValueType::get(parser.getBuilder().getContext()))))
    {
      return failure();
    }

    if (attrs.has_value())
    {
      if (parser.parseOptionalAttrDictWithKeyword(attrs->get()))
      {
        return failure();
      }
    }

    assert(argNames.size() == argTypes.size());

    return parser.parseRegion(region, argNames, argTypes);
  }

  void printVeronaBody(
    mlir::OpAsmPrinter& printer,
    mlir::Region& region,
    mlir::DictionaryAttr attrs,
    llvm::ArrayRef<llvm::StringRef> elidedAttrs)
  {
    printArgumentList(
      printer, region.getArguments(), ValueType::get(region.getContext()));
    if (attrs)
    {
      printer.printOptionalAttrDictWithKeyword(attrs.getValue(), elidedAttrs);
    }
    printer.printRegion(
      region, /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/true);
  }

  /*
  void printOp(OpAsmPrinter& printer, MethodOp op)
  {
    printer << MethodOp::getOperationName() << " " << op.nameAttr() << ":" <<
  op.selector() << " XX "; printVeronaBody(printer, op, op.body(),
  op->getAttrDictionary(), { "name", "selector" });
  }

  ParseResult parseMethodOp(OpAsmParser& parser, OperationState& state)
  {
    SmallVector<OpAsmParser::OperandType, 4> argNames;
    SmallVector<Type, 4> argTypes;
    auto& builder = parser.getBuilder();

    StringAttr name;
    IntegerAttr selector;
    if (parser.parseAttribute(
          name,
          parser.getBuilder().getType<::mlir::NoneType>(),
          "name",
          state.attributes))
      return failure();
    if (parser.parseColon())
      return failure();
    if (parser.parseAttribute(
          selector,
          parser.getBuilder().getIntegerType(32),
          "selector",
          state.attributes))
      return failure();

    if (failed(parseVeronaBody(parser, *state.addRegion(), state.attributes)))
      return failure();

    return success();
  }
  */
}
