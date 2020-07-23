#pragma once

#include "mlir/IR/OpDefinition.h"

namespace mlir::verona
{
  struct PolymorphicOp
  {
    static StringRef getTypeParametersAttrName()
    {
      return "type_parameters";
    }

    static ParseResult parseTypeParameters(OpAsmParser& parser, OperationState& state);
    static void printTypeParameters(OpAsmPrinter& printer, Operation* op);
    static LogicalResult verifyOperation(Operation* op);
    static size_t getNumTypeParameters(Operation* op);
    static Type getTypeParameterBound(Operation* op, size_t index);
    private:
    static ArrayAttr getAttribute(Operation* op);
  };
}
