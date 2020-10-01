// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "Query.h"
#include "dialect/Typechecker.h"
#include "dialect/VeronaTypes.h"

#include "llvm/ADT/SetVector.h"

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

  struct RegionRelationship
  {
    static std::optional<RegionRelationship> fromFieldType(Type type)
    {
      assert(isaVeronaType(type));

      MLIRContext* ctx = type.getContext();
      if (isSubtype(type, getWritable(ctx)))
      {
        bool isIso = isSubtype(type, getIso(ctx));
        bool isMut = isSubtype(type, getMut(ctx));
        return RegionRelationship{isIso, isMut};
      }
      else
      {
        return std::nullopt;
      }
    }

    static RegionRelationship top()
    {
      return RegionRelationship(true, true);
    }

    static RegionRelationship bottom()
    {
      return RegionRelationship(true, true);
    }

    static RegionRelationship mut()
    {
      return RegionRelationship(false, true);
    }

    RegionRelationship concat(RegionRelationship other) const
    {
      return RegionRelationship(
        isStrictSubRegion || other.isStrictSubRegion,
        isSameRegion && other.isSameRegion);
    }

    RegionRelationship join(RegionRelationship other) const
    {
      return RegionRelationship(
        isStrictSubRegion && other.isStrictSubRegion,
        isSameRegion && other.isSameRegion);
    }

    RegionRelationship meet(RegionRelationship other) const
    {
      return RegionRelationship(
        isStrictSubRegion || other.isStrictSubRegion,
        isSameRegion || other.isSameRegion);
    }

    bool operator<(RegionRelationship other) const
    {
      return std::tie(isStrictSubRegion, isSameRegion) <
        std::tie(other.isStrictSubRegion, other.isSameRegion);
    }

    friend llvm::raw_ostream&
    operator<<(llvm::raw_ostream& os, RegionRelationship self)
    {
      return os << self.isStrictSubRegion << " " << self.isSameRegion;
    }

  private:
    RegionRelationship(bool isStrictSubRegion, bool isSameRegion)
    : isStrictSubRegion(isStrictSubRegion), isSameRegion(isSameRegion)
    {}

    bool isStrictSubRegion;
    bool isSameRegion;
  };

  /// The region of `left` is the same as or a child of the region of `right`.
  /// The precise nature of the relationship between `left` and `right` depends
  /// on `types`:
  ///
  /// - If all elements in `types` are subtypes of mut, left and right are in
  ///   the same region. This includes the case where `types` is empty.
  ///
  /// - If at least one element in `type` is a subtype of iso, `left` is in a
  ///   strict sub-region of `right`'s region.
  ///
  /// - Otherwise the relationship is unknown, and we can only assume that
  ///   `left`'s region is somewhere in the region tree dominated by `right`.
  struct In
  {
    In(Value left, Value right, RegionRelationship relationship)
    : left(left), right(right), relationship(relationship)
    {}

    Value left;
    Value right;
    RegionRelationship relationship;

    auto data() const
    {
      return std::tie(left, right, relationship);
    }

    void print(llvm::raw_ostream& os, AsmState& state) const
    {
      os << "in(";
      Value(left).printAsOperand(os, state);
      os << ", ";
      Value(right).printAsOperand(os, state);
      os << ", " << relationship << ")";
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
