#pragma once

#include "mlir/IR/OpDefinition.h"

namespace mlir::verona
{
  struct VariableType;
  struct PolymorphicOp
  {
    static StringRef getTypeParametersAttrName()
    {
      return "type_parameters";
    }

    static ParseResult
    parseTypeParameters(OpAsmParser& parser, OperationState& state);
    static void printTypeParameters(OpAsmPrinter& printer, Operation* op);
    static LogicalResult verifyOperation(Operation* op);
    static size_t getNumTypeParameters(Operation* op);
    static Type getTypeParameterBound(Operation* op, size_t index);
    static Type getTypeVariableBound(Operation* op, VariableType var);
    static Type replaceTypeVariables(Type type, ArrayRef<Type> values);
    static void replaceTypeVariables(
      ArrayRef<Type> types,
      ArrayRef<Type> values,
      SmallVectorImpl<Type>& result);

  private:
    static ArrayAttr getAttribute(Operation* op);
  };
}
