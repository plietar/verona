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

using namespace mlir;

static ParseResult parseClassOp(OpAsmParser& parser, OperationState& state)
{
  Region* body = state.addRegion();

  StringAttr nameAttr;
  if (parser.parseSymbolName(
        nameAttr, SymbolTable::getSymbolAttrName(), state.attributes))
    return failure();

  if (verona::PolymorphicOp::parseTypeParameters(parser, state))
    return failure();

  if (parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return failure();

  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  verona::ClassOp::ensureTerminator(*body, parser.getBuilder(), state.location);

  return success();
}

static void print(OpAsmPrinter& printer, verona::ClassOp op)
{
  printer << verona::ClassOp::getOperationName() << ' ';
  printer.printSymbolName(op.sym_name());
  verona::PolymorphicOp::printTypeParameters(printer, op);
  printer.printOptionalAttrDict(
    op.getAttrs(),
    /*elidedAttrs=*/
    {SymbolTable::getSymbolAttrName(),
     verona::PolymorphicOp::getTypeParametersAttrName()});
  printer.printRegion(
    op.body(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

/**
 * AllocateRegionOp and AllocateObjectOp share similar features, which we verify
 * here:
 * - The class referenced by the operation must exist.
 * - There must be as many field names as there are operands.
 * - The set of field names must match those specified in the class definition
 *   (TODO: not implemented yet)
 */
template<typename Op>
static LogicalResult verifyAllocationOp(Op op)
{
  auto className = op.class_name();
  auto classOp =
    SymbolTable::lookupNearestSymbolFrom<verona::ClassOp>(op, className);
  if (!classOp)
  {
    return op.emitOpError("class '")
      << className << "' not found in nearest symbol table";
  }

  if (op.field_names().size() != op.fields().size())
  {
    return op.emitError("The number of operands (")
      << op.fields().size() << ") for '" << op.getOperationName()
      << "' op does not match the number of field names ("
      << op.field_names().size() << ")";
  }

  return success();
}

static LogicalResult verify(verona::AllocateRegionOp op)
{
  return verifyAllocationOp(op);
}

static LogicalResult verify(verona::AllocateObjectOp op)
{
  return verifyAllocationOp(op);
}

static LogicalResult verify(verona::ReturnOp op)
{
  auto fn = cast<FuncOp>(op.getParentOp());

  // Only check that the number of operands matches the function signature.
  // Actual typechecking of the operands is done in the `typecheck` pass.
  const auto& results = fn.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError("has ")
      << op.getNumOperands() << " operands, but enclosing function (@"
      << fn.getName() << ") returns " << results.size();

  return success();
}

static LogicalResult verify(verona::ClassOp classOp)
{
  Block* body = classOp.getBody();

  // Verify that classes contain only fields, and that there are no duplicates.
  // TODO: once we add methods, verify that there are no duplicates either.
  llvm::StringSet<> fields;
  for (Operation& op : *body)
  {
    if (verona::FieldOp field_op = dyn_cast<verona::FieldOp>(op))
    {
      auto it = fields.insert(field_op.name());

      if (!it.second)
      {
        return op.emitError().append(
          "redefinition of field named '", field_op.name(), "'");
      }
    }
    else if (!isa<verona::ClassEndOp>(op))
    {
      return op.emitOpError("cannot be contained in class");
    }
  }

  return success();
}

namespace mlir::verona
{
  LogicalResult CopyOp::typecheck()
  {
    return checkSubtype(getOperation(), input().getType(), output().getType());
  }

  LogicalResult FieldReadOp::typecheck()
  {
    auto originType = origin().getType();

    auto [fieldType, _] =
      lookupFieldType(getOperation(), originType, field());

    auto adaptedType = ViewpointType::get(getContext(), originType, fieldType);

    if (!fieldType)
      return emitOpError("Cannot find field '")
        << field() << "' in type " << origin().getType();

    return checkSubtype(getOperation(), adaptedType, output().getType());
  }

  LogicalResult FieldWriteOp::typecheck()
  {
    auto [readType, writeType] =
      lookupFieldType(getOperation(), origin().getType(), field());

    assert((readType == nullptr) == (writeType == nullptr));
    if (!readType)
      return emitOpError("Cannot find field '")
        << field() << "' in type " << origin().getType();

    if (failed(checkSubtype(getOperation(), value().getType(), writeType)))
      return failure();
    return checkSubtype(getOperation(), readType, output().getType());
  }

  LogicalResult ReturnOp::typecheck()
  {
    auto fn = cast<FuncOp>(getParentOp());
    const auto& results = fn.getType().getResults();

    // The verifier already checks that the number of operands matches the
    // function signature. We only need to check types.
    for (auto [operand, type] : llvm::zip(operands(), results))
    {
      if (failed(checkSubtype(getOperation(), operand.getType(), type)))
        return failure();
    }

    return success();
  }

#define GET_OP_CLASSES
#include "dialect/VeronaOps.cpp.inc"

} // namespace mlir::verona
