// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "Query.h"
#include "dialect/VeronaTypes.h"

/// Topological Facts are a static approximation on the runtime shape of the
/// heap.
///
/// We use these facts to determine whether certain operations are safe. For
/// example, in the statement `x.f = y` (where f is a mut field), we require
/// that x and y are in the same region, or in other words, `In(y, x, nil)`.
namespace mlir::verona
{
  /// Values `left` and `right` point the the same object.
  struct Alias
  {
    Alias(Value left, Value right) : left(left), right(right) {}

    Value left;
    Value right;

    auto data() const
    {
      return std::tie(left, right);
    }

    void print(llvm::raw_ostream& os, AsmState& state) const
    {
      os << "alias(";
      Value(left).printAsOperand(os, state);
      os << ", ";
      Value(right).printAsOperand(os, state);
      os << ")";
    }
  };

  /// This fact is true in basic blocks that define the Value. Not really a
  /// topological fact per-se, but useful to implement rules that range over all
  /// variables (eg. reflexivity).
  struct Defined
  {
    Defined(Value value) : value(value)
    {
      assert(isaVeronaType(value.getType()));
    }

    Value value;

    auto data() const
    {
      return std::tie(value);
    }

    void print(llvm::raw_ostream& os, AsmState& state) const
    {
      os << "defined(";
      Value(value).printAsOperand(os, state);
      os << " : " << value.getType() << ")";
    }
  };

}

namespace llvm
{
  template<>
  struct DenseMapInfo<mlir::verona::Alias>
  {
    static inline mlir::verona::Alias getEmptyKey()
    {
      return mlir::verona::Alias(
        DenseMapInfo<mlir::Value>::getEmptyKey(),
        DenseMapInfo<mlir::Value>::getEmptyKey());
    }

    static inline mlir::verona::Alias getTombstoneKey()
    {
      return mlir::verona::Alias(
        DenseMapInfo<mlir::Value>::getTombstoneKey(),
        DenseMapInfo<mlir::Value>::getTombstoneKey());
    }

    static unsigned getHashValue(const mlir::verona::Alias& value)
    {
      return hash_value(value.data());
    }

    static bool
    isEqual(const mlir::verona::Alias& lhs, const mlir::verona::Alias& rhs)
    {
      return lhs.data() == rhs.data();
    }
  };
}
