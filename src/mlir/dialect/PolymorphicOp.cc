#include "dialect/PolymorphicOp.h"

#include "dialect/VeronaTraits.h"
#include "dialect/VeronaTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

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

  Type PolymorphicOp::getTypeVariableBound(Operation* op, VariableType var)
  {
    size_t index = var.getIndex();
    while (true)
    {
      if (op->hasTrait<OpTrait::PolymorphicOpTrait>())
      {
        size_t count = PolymorphicOp::getNumTypeParameters(op);
        if (index >= count)
          index -= count;
        else
          break;
      }

      op = op->getParentOp();
    }

    return getTypeParameterBound(op, index);
  }

  Type PolymorphicOp::replaceTypeVariables(Type type, ArrayRef<Type> values)
  {
    MLIRContext* ctx = type.getContext();
    assert(isaVeronaType(type));
    switch (type.getKind())
    {
      case VeronaTypes::Integer:
      case VeronaTypes::Capability:
        return type;

      case VeronaTypes::Variable:
        assert(type.cast<VariableType>().getIndex() < values.size());
        return values[type.cast<VariableType>().getIndex()];

      case VeronaTypes::Meet:
      {
        MeetType meet = type.cast<MeetType>();
        SmallVector<Type, 4> elements;
        replaceTypeVariables(meet.getElements(), values, elements);
        return MeetType::get(ctx, elements);
      }

      case VeronaTypes::Join:
      {
        JoinType join = type.cast<JoinType>();
        SmallVector<Type, 4> elements;
        replaceTypeVariables(join.getElements(), values, elements);
        return JoinType::get(ctx, elements);
      }

      case VeronaTypes::Class:
      {
        ClassType classType = type.cast<ClassType>();
        SmallVector<Type, 4> arguments;
        replaceTypeVariables(classType.getArguments(), values, arguments);
        return ClassType::get(ctx, classType.getClassName(), arguments);
      }

      default:
        abort();
    }
  }

  void PolymorphicOp::replaceTypeVariables(
    ArrayRef<Type> types, ArrayRef<Type> values, SmallVectorImpl<Type>& result)
  {
    llvm::transform(types, std::back_inserter(result), [&](Type type) {
      return replaceTypeVariables(type, values);
    });
  }
}
