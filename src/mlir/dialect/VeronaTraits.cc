#include "VeronaTraits.h"

#include "VeronaTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::verona
{
  LogicalResult
  verifyTypeUsage(Operation* op, Type type, const char* description)
  {
    if (freeTypeVariables(type) > typeVariablesInScope(op))
      return op->emitError()
        << description << " " << type << " refers to unbound type variables";
    else
      return success();
  }

  LogicalResult verifyTypeUses(Operation* op)
  {
    for (Type type : op->getOperandTypes())
    {
      if (failed(verona::verifyTypeUsage(op, type, "Operand type")))
        return failure();
    }

    for (Type type : op->getResultTypes())
    {
      if (failed(verona::verifyTypeUsage(op, type, "Result type")))
        return failure();
    }

    for (auto& [_, attr] : op->getAttrs())
    {
      if (TypeAttr typeAttr = attr.dyn_cast<TypeAttr>())
      {
        if (failed(
              verona::verifyTypeUsage(op, typeAttr.getValue(), "Result type")))
          return failure();
      }
    }

    return success();
  }
}
