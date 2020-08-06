// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "VeronaOps.h"
#include "mlir/IR/Dialect.h"

namespace mlir::verona
{
  Type parseVeronaType(DialectAsmParser& parser);
  void printVeronaType(Type type, DialectAsmPrinter& os);

  /// Returns true if the type is one defined by the Verona dialect.
  bool isaVeronaType(Type type);
  /// Returns true if all types in the array are ones defined by the Verona
  /// dialect.
  bool areVeronaTypes(llvm::ArrayRef<Type> types);

  /// Normalize a type by distributing unions and intersections, putting the
  /// type in disjunctive normal form. This is a necessary step in order for
  /// subtyping to recognise certain relations.
  ///
  /// The amount of normalization done is quite limited. In particular it does
  /// not do any flattening (eg. `join<A, join<B, C>>` into `join<A, B, C>`),
  /// nor does it do any simplification (eg. `join<A, A, B>` into `join<A, B>`).
  ///
  /// TODO: normalizing types is a potentially expensive operation, so we should
  /// try to cache the results.
  Type normalizeType(Type type);

  FieldOp lookupClassField(ClassOp classOp, StringRef name);

  std::pair<Type, Type>
  lookupFieldType(Operation* op, Type origin, StringRef name);

  /// Look up the type of a field in many `origins` types.
  ///
  /// The read sides are collected into `readElements` and the write sides into
  /// `writeElements`.
  ///
  /// Returns true if the field is found in every element of `origins`. This
  /// includes returning true is `origins` is empty. If false is returned, at
  /// least one origin does not expose this field, and
  /// `readElements`/`writeElements` will be smaller in length than `origins`.
  ///
  /// `readElements` and `writeElements` always have the same length.
  ///
  bool lookupFieldTypes(
    Operation* op,
    ArrayRef<Type> origins,
    StringRef name,
    SmallVectorImpl<Type>& readElements,
    SmallVectorImpl<Type>& writeElements);

  namespace detail
  {
    struct MeetTypeStorage;
    struct JoinTypeStorage;
    struct IntegerTypeStorage;
    struct CapabilityTypeStorage;
    struct ClassTypeStorage;
    struct ViewpointTypeStorage;
  }

  // In the long term we should claim a range in LLVM's DialectSymbolRegistry,
  // rather than use the "experimental" range.
  static constexpr unsigned FIRST_VERONA_TYPE =
    Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE;
  static constexpr unsigned LAST_VERONA_TYPE =
    Type::Kind::LAST_PRIVATE_EXPERIMENTAL_0_TYPE;

  namespace VeronaTypes
  {
    enum Kind
    {
      Meet = FIRST_VERONA_TYPE,
      Join,
      Integer,
      Capability,
      Class,
      Viewpoint,
    };
  }

  struct MeetType
  : public Type::TypeBase<MeetType, Type, detail::MeetTypeStorage>
  {
    using Base::Base;
    static MeetType
    get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elementTypes);
    llvm::ArrayRef<mlir::Type> getElements() const;

    static bool kindof(unsigned kind)
    {
      return kind == VeronaTypes::Meet;
    }
  };

  struct JoinType
  : public Type::TypeBase<JoinType, Type, detail::JoinTypeStorage>
  {
    using Base::Base;
    static JoinType
    get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elementTypes);
    llvm::ArrayRef<mlir::Type> getElements() const;

    static bool kindof(unsigned kind)
    {
      return kind == VeronaTypes::Join;
    }
  };

  struct IntegerType
  : public Type::TypeBase<IntegerType, Type, detail::IntegerTypeStorage>
  {
    using Base::Base;

    static IntegerType get(MLIRContext* context, size_t width, unsigned sign);

    size_t getWidth() const;
    bool getSign() const;

    static bool kindof(unsigned kind)
    {
      return kind == VeronaTypes::Integer;
    }
  };

  enum class Capability
  {
    Isolated,
    Mutable,
    Immutable,
  };

  struct CapabilityType
  : public Type::TypeBase<CapabilityType, Type, detail::CapabilityTypeStorage>
  {
    using Base::Base;
    static CapabilityType get(MLIRContext* ctx, Capability cap);
    Capability getCapability() const;

    static bool kindof(unsigned kind)
    {
      return kind == VeronaTypes::Capability;
    }
  };

  struct ClassType
  : public Type::TypeBase<ClassType, Type, detail::ClassTypeStorage>
  {
    using Base::Base;
    static ClassType get(MLIRContext* ctx, StringRef s);
    StringRef getClassName() const;

    static bool kindof(unsigned kind)
    {
      return kind == VeronaTypes::Class;
    }
  };

  struct ViewpointType
  : public Type::TypeBase<ViewpointType, Type, detail::ViewpointTypeStorage>
  {
    using Base::Base;
    static ViewpointType get(MLIRContext* ctx, Type left, Type right);

    Type getLeftType() const;
    Type getRightType() const;

    static bool kindof(unsigned kind)
    {
      return kind == VeronaTypes::Viewpoint;
    }
  };

  // Various convenience functions used to construct commonly used Verona types.
  // TODO: These should be constructed upfront and cached in some context
  // object.
  inline Type getIso(MLIRContext* ctx)
  {
    return CapabilityType::get(ctx, Capability::Isolated);
  }
  inline Type getMut(MLIRContext* ctx)
  {
    return CapabilityType::get(ctx, Capability::Mutable);
  }
  inline Type getImm(MLIRContext* ctx)
  {
    return CapabilityType::get(ctx, Capability::Immutable);
  }
  inline Type getWritable(MLIRContext* ctx)
  {
    return JoinType::get(ctx, {getIso(ctx), getMut(ctx)});
  }
  inline Type getAnyCapability(MLIRContext* ctx)
  {
    return JoinType::get(ctx, {getIso(ctx), getMut(ctx), getImm(ctx)});
  }
}
