#include "dialect/PolymorphicOp.h"

#include "mlir/IR/OpImplementation.h"
#include "dialect/VeronaTraits.h"
#include "mlir/IR/Builders.h"
#include "dialect/VeronaTypes.h"

namespace mlir::verona
{
  static ParseResult
  parseTypeListAttr(OpAsmParser& parser, SmallVectorImpl<Attribute>& attr)
  {
    if (parser.parseOptionalLSquare())
      return success();

    if (succeeded(parser.parseOptionalRSquare()))
      return success();

    do
    {
      Type element;
      if (parser.parseType(element))
        return failure();

      attr.push_back(TypeAttr::get(element));
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare())
      return failure();

    return success();
  }

  ParseResult
  PolymorphicOp::parseTypeParameters(OpAsmParser& parser, OperationState& state)
  {
    MLIRContext* ctx = parser.getBuilder().getContext();

    SmallVector<Attribute, 4> attrs;
    if (parseTypeListAttr(parser, attrs))
      return failure();

    state.attributes.append(
      getTypeParametersAttrName(), ArrayAttr::get(attrs, ctx));
    return success();
  }

  void PolymorphicOp::printTypeParameters(OpAsmPrinter& printer, Operation* op)
  {
    auto attr = getAttribute(op);
    if (!attr.empty())
      printer.printAttribute(attr);
  }

  LogicalResult PolymorphicOp::verifyOperation(Operation* op)
  {
    assert(op->hasTrait<OpTrait::PolymorphicOpTrait>());

    StringRef attrName = getTypeParametersAttrName();
    auto attr = op->getAttr(attrName);
    if (!attr)
      return op->emitOpError("requires attribute '") << attrName << "'";
    if (!attr.isa<ArrayAttr>())
      return op->emitError("attribute '") << attrName << "' must be an array";

    auto verifyTypeParameter = [](Attribute inner) -> bool {
      return inner.isa<TypeAttr>() &&
        isaVeronaType(inner.cast<TypeAttr>().getValue());
    };
    if (!llvm::all_of(attr.cast<ArrayAttr>(), verifyTypeParameter))
    {
      return op->emitError("elements of attribute '")
        << attrName << "' must be Verona types";
    }

    return success();
  }

  size_t PolymorphicOp::getNumTypeParameters(Operation* op)
  {
    return getAttribute(op).size();
  }

  Type PolymorphicOp::getTypeParameterBound(Operation* op, size_t index)
  {
    ArrayAttr arrayAttr = getAttribute(op);
    assert(index < arrayAttr.size());
    return arrayAttr[index].cast<TypeAttr>().getValue();
  }

  ArrayAttr PolymorphicOp::getAttribute(Operation* op)
  {
    assert(op->hasTrait<OpTrait::PolymorphicOpTrait>());
    auto attr = op->getAttrOfType<ArrayAttr>(getTypeParametersAttrName());
    assert(attr != nullptr);
    return attr;
  }
}
