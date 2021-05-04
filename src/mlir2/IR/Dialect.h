#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir2/IR/Attributes.h"
#include "mlir2/IR/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir2/IR/TypeDefs.h.inc"

#define GET_OP_CLASSES
#include "mlir2/IR/Ops.h.inc"
