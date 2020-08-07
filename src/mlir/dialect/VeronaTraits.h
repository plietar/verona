#pragma once

#include "dialect/PolymorphicOp.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::verona
{
  LogicalResult
  verifyTypeUsage(Operation* op, Type type, const char* description);

  LogicalResult verifyTypeUses(Operation* op);
}

namespace mlir::OpTrait
{
  template<typename ConcreteType>
  class VerifyTypesTrait : public TraitBase<ConcreteType, VerifyTypesTrait>
  {
  public:
    static LogicalResult verifyTrait(Operation* op)
    {
      return verona::verifyTypeUses(op);
    }
  };

  template<typename ConcreteType>
  class PolymorphicOpTrait : public TraitBase<ConcreteType, PolymorphicOpTrait>
  {
  public:
    static LogicalResult verifyTrait(Operation* op)
    {
      return verona::PolymorphicOp::verifyOperation(op);
    }
  };
}
