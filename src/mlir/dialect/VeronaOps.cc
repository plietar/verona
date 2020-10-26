// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "VeronaOps.h"

#include "Typechecker.h"
#include "VeronaDialect.h"
#include "VeronaTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/ADT/StringSet.h"

namespace mlir::verona
{
  /**
   * AllocateRegionOp and AllocateObjectOp share similar features, which we
   * verify here:
   * - There must be as many field names as there are operands.
   * - The set of field names must match those specified in the class type
   *   (TODO: not implemented yet)
   */
  template<typename Op>
  static LogicalResult verifyAllocationOp(Op op)
  {
    if (op.field_names().size() != op.fields().size())
    {
      return op.emitError("The number of operands (")
        << op.fields().size() << ") for '" << op.getOperationName()
        << "' op does not match the number of field names ("
        << op.field_names().size() << ")";
    }

    return success();
  }

  static LogicalResult verify(AllocateRegionOp op)
  {
    return verifyAllocationOp(op);
  }

  static LogicalResult verify(AllocateObjectOp op)
  {
    return verifyAllocationOp(op);
  }

  static LogicalResult verify(MethodOp op)
  {
    MLIRContext* ctx = op.getContext();
    Type metaType = MetaType::get(ctx);
    Type metaValue = ValueType::get(ctx);

    if (op.signature().getBlocks().size() != 1)
      return op.emitError("Signature region must have exactly one region");
    if (op.body().empty())
      return op.emitError("Body region must have at least one region");

    for (size_t i = 0; i < op.signature().getNumArguments(); i++)
    {
      Value arg = op.signature().getArgument(i);
      if (arg.getType() != metaType)
      {
        return op.emitError("Argument ")
          << i << " of signature block must be `!verona.type`. Found "
          << arg.getType() << " instead.";
      }
    }

    if (op.body().getNumArguments() < op.signature().getNumArguments())
    {
      return op.emitError(
        "Body block must have at least as many arguments as the signature "
        "block does.");
    }

    for (size_t i = 0; i < op.signature().getNumArguments(); i++)
    {
      Value arg = op.body().getArgument(i);
      if (arg.getType() != metaType)
      {
        return op.emitError("Argument ")
          << i << " of body block must be " << metaType << ". Found "
          << arg.getType() << " instead.";
      }
    }

    for (size_t i = op.signature().getNumArguments();
         i < op.body().getNumArguments();
         i++)
    {
      Value arg = op.body().getArgument(i);
      if (arg.getType() != metaValue)
      {
        return op.emitError("Argument ")
          << i << " of body block must be " << metaValue << ". Found "
          << arg.getType() << " instead.";
      }
    }

    return success();
  }

  void print(OpAsmPrinter& p, MethodOp op)
  {
    p << op.getOperationName() << ' ';
    p.printSymbolName(op.name());
    p.printOptionalAttrDictWithKeyword(op.getAttrs(), {"name"});

    p << " {";
    p << " [";
    for (size_t i = 0; i < op.signature().getNumArguments(); i++)
    {
      if (i > 0)
        p << ", ";
      p.printOperand(op.signature().getArgument(i));
    }
    p << "]";

    p.printRegion(op.signature(), false, true);

    p << " [";
    for (size_t i = 0; i < op.signature().getNumArguments(); i++)
    {
      if (i > 0)
        p << ", ";
      p.printOperand(op.body().getArgument(i));
    }
    p << "]";
    p << "(";
    for (size_t i = op.signature().getNumArguments();
         i < op.body().getNumArguments();
         i++)
    {
      if (i > op.signature().getNumArguments())
        p << ", ";
      p.printOperand(op.body().getArgument(i));
    }
    p << ")";

    p.printRegion(op.body(), false, true);
    p << " }";
  }

  void print(OpAsmPrinter& p, ClassOp op)
  {
    p << op.getOperationName() << ' ';
    p.printSymbolName(op.name());
    p.printOptionalAttrDictWithKeyword(op.getAttrs(), {"name"});

    p << " [";
    for (size_t i = 0; i < op.body().getNumArguments(); i++)
    {
      if (i > 0)
        p << ", ";
      p.printOperand(op.body().getArgument(i));
    }
    p << "]";

    p.printRegion(op.body(), false, false);
  }

  ParseResult parseParams(
    OpAsmParser& parser,
    OpAsmParser::Delimiter delimiter,
    Type type,
    SmallVectorImpl<OpAsmParser::OperandType>& values,
    SmallVectorImpl<Type>& types)
  {
    assert(values.size() == types.size());

    if (parser.parseRegionArgumentList(values, delimiter))
      return failure();

    types.resize(values.size(), type);
    return success();
  }

  ParseResult parseParametricRegion(
    OpAsmParser& parser,
    OperationState& result,
    bool hasValueParams,
    Region** pregion = nullptr)
  {
    using Delimiter = OpAsmParser::Delimiter;

    MLIRContext* ctx = parser.getBuilder().getContext();
    Type metaType = MetaType::get(ctx);
    Type metaValue = ValueType::get(ctx);

    SmallVector<OpAsmParser::OperandType, 4> operands;
    SmallVector<Type, 4> types;

    Region* region = result.addRegion();

    if (parseParams(parser, Delimiter::Square, metaType, operands, types))
      return failure();

    if (hasValueParams)
    {
      if (parseParams(parser, Delimiter::Paren, metaValue, operands, types))
        return failure();
    }

    if (parser.parseRegion(*region, operands, types))
      return failure();

    if (pregion != nullptr)
      *pregion = region;

    return success();
  }

  ParseResult parseMethodOp(OpAsmParser& parser, OperationState& result)
  {
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, "name", result.attributes))
      return failure();

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
      return failure();

    if (parser.parseLBrace())
      return failure();

    if (parseParametricRegion(parser, result, false))
      return failure();

    if (parseParametricRegion(parser, result, true))
      return failure();

    if (parser.parseRBrace())
      return failure();

    return success();
  }

  ParseResult parseClassOp(OpAsmParser& parser, OperationState& result)
  {
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, "name", result.attributes))
      return failure();

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
      return failure();

    Region* body;
    if (parseParametricRegion(parser, result, false, &body))
      return failure();

    ClassOp::ensureTerminator(*body, parser.getBuilder(), result.location);

    return success();
  }

  Type FieldReadOp::getFieldType()
  {
    return lookupFieldType(origin().getType(), field()).first;
  }

  std::pair<Type, Type> FieldWriteOp::getFieldType()
  {
    return lookupFieldType(origin().getType(), field());
  }
}

#define GET_OP_CLASSES
#include "dialect/VeronaOps.cpp.inc"
