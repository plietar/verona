// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "dialect/VeronaTypes.h"

#include "dialect/VeronaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir::verona::detail
{
  struct MeetTypeStorage : public TypeStorage
  {
    using KeyTy = llvm::ArrayRef<Type>;

    llvm::ArrayRef<Type> elements;

    MeetTypeStorage(llvm::ArrayRef<Type> elements) : elements(elements) {}

    bool operator==(const KeyTy& key) const
    {
      return key == elements;
    }

    static MeetTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      llvm::ArrayRef<mlir::Type> elements = allocator.copyInto(key);
      return new (allocator.allocate<MeetTypeStorage>())
        MeetTypeStorage(elements);
    }
  };

  struct JoinTypeStorage : public TypeStorage
  {
    using KeyTy = llvm::ArrayRef<Type>;

    llvm::ArrayRef<Type> elements;

    JoinTypeStorage(llvm::ArrayRef<Type> elements) : elements(elements) {}

    bool operator==(const KeyTy& key) const
    {
      return key == elements;
    }

    static JoinTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      llvm::ArrayRef<mlir::Type> elements = allocator.copyInto(key);
      return new (allocator.allocate<JoinTypeStorage>())
        JoinTypeStorage(elements);
    }
  };

  struct IntegerTypeStorage : public ::mlir::TypeStorage
  {
    uint8_t width;
    enum SignType
    {
      Unknown,
      Unsigned,
      Signed
    };
    unsigned sign;

    // width, sign
    using KeyTy = std::tuple<size_t, unsigned>;
    IntegerTypeStorage(const KeyTy& key)
    : width(std::get<0>(key)), sign(std::get<1>(key))
    {}

    bool operator==(const KeyTy& key) const
    {
      return key == KeyTy(width, sign);
    }

    static IntegerTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      return new (allocator.allocate<IntegerTypeStorage>())
        IntegerTypeStorage(key);
    }
  };

  struct CapabilityTypeStorage : public ::mlir::TypeStorage
  {
    Capability capability;

    using KeyTy = Capability;

    CapabilityTypeStorage(const KeyTy& key) : capability(key) {}

    static llvm::hash_code hashKey(const KeyTy& key)
    {
      return llvm::hash_value(key);
    }

    bool operator==(const KeyTy& key) const
    {
      return key == capability;
    }

    static CapabilityTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      return new (allocator.allocate<CapabilityTypeStorage>())
        CapabilityTypeStorage(key);
    }
  };

  struct ClassTypeStorage : public ::mlir::TypeStorage
  {
    std::string class_name;
    ArrayRef<Type> arguments;

    using KeyTy = std::tuple<StringRef, ArrayRef<Type>>;

    ClassTypeStorage(StringRef class_name, ArrayRef<Type> arguments)
    : class_name(class_name), arguments(arguments)
    {}

    static llvm::hash_code hashKey(const KeyTy& key)
    {
      return llvm::hash_value(key);
    }

    bool operator==(const KeyTy& key) const
    {
      return key == std::tie(class_name, arguments);
    }

    static ClassTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      StringRef class_name = allocator.copyInto(std::get<0>(key));
      ArrayRef<mlir::Type> arguments = allocator.copyInto(std::get<1>(key));
      return new (allocator.allocate<ClassTypeStorage>())
        ClassTypeStorage(class_name, arguments);
    }
  };

  struct ViewpointTypeStorage : public ::mlir::TypeStorage
  {
    Type left;
    Type right;

    using KeyTy = std::tuple<Type, Type>;

    ViewpointTypeStorage(Type left, Type right) : left(left), right(right) {}

    static llvm::hash_code hashKey(const KeyTy& key)
    {
      return llvm::hash_value(key);
    }

    bool operator==(const KeyTy& key) const
    {
      return key == std::tie(left, right);
    }

    static ViewpointTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      return new (allocator.allocate<ViewpointTypeStorage>())
        ViewpointTypeStorage(std::get<0>(key), std::get<1>(key));
    }
  };

  struct VariableTypeStorage : public ::mlir::TypeStorage
  {
    uint64_t index;

    using KeyTy = uint64_t;
    VariableTypeStorage(uint64_t index) : index(index) {}

    bool operator==(const KeyTy& key) const
    {
      return key == index;
    }

    static VariableTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      return new (allocator.allocate<VariableTypeStorage>())
        VariableTypeStorage(key);
    }
  };
} // namespace mlir::verona::detail

namespace mlir::verona
{
  MeetType MeetType::get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elements)
  {
    assert(areVeronaTypes(elements));
    return Base::get(ctx, VeronaTypes::Meet, elements);
  }

  llvm::ArrayRef<mlir::Type> MeetType::getElements() const
  {
    return getImpl()->elements;
  }

  JoinType JoinType::get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elements)
  {
    assert(areVeronaTypes(elements));
    return Base::get(ctx, VeronaTypes::Join, elements);
  }

  llvm::ArrayRef<mlir::Type> JoinType::getElements() const
  {
    return getImpl()->elements;
  }

  IntegerType IntegerType::get(MLIRContext* ctx, size_t width, unsigned sign)
  {
    return Base::get(ctx, VeronaTypes::Integer, width, sign);
  }

  size_t IntegerType::getWidth() const
  {
    return getImpl()->width;
  }

  bool IntegerType::getSign() const
  {
    return getImpl()->sign;
  }

  CapabilityType CapabilityType::get(MLIRContext* ctx, Capability cap)
  {
    return Base::get(ctx, VeronaTypes::Capability, cap);
  }

  Capability CapabilityType::getCapability() const
  {
    return getImpl()->capability;
  }

  ClassType ClassType::get(
    MLIRContext* ctx, StringRef class_name, ArrayRef<Type> arguments)
  {
    return Base::get(ctx, VeronaTypes::Class, class_name, arguments);
  }

  StringRef ClassType::getClassName() const
  {
    return getImpl()->class_name;
  }

  ArrayRef<Type> ClassType::getArguments() const
  {
    return getImpl()->arguments;
  }

  ViewpointType ViewpointType::get(MLIRContext* ctx, Type left, Type right)
  {
    return Base::get(ctx, VeronaTypes::Viewpoint, left, right);
  }

  Type ViewpointType::getLeftType() const
  {
    return getImpl()->left;
  }

  Type ViewpointType::getRightType() const
  {
    return getImpl()->right;
  }

  VariableType VariableType::get(MLIRContext* ctx, uint64_t index)
  {
    return Base::get(ctx, VeronaTypes::Variable, index);
  }

  uint64_t VariableType::getIndex() const
  {
    return getImpl()->index;
  }

  /// Parse a list of types, surrounded by angle brackets and separated by
  /// commas. The types inside the list must be Verona types and should not use
  /// the `!verona.` prefix.
  ///
  /// Empty lists are allowed, but must still use angle brackets, i.e. `< >`.
  /// Lists of one elements are also allowed.
  static ParseResult
  parseTypeList(DialectAsmParser& parser, llvm::SmallVectorImpl<Type>& result)
  {
    if (parser.parseLess())
      return failure();

    if (succeeded(parser.parseOptionalGreater()))
      return success();

    do
    {
      mlir::Type element = parseVeronaType(parser);
      if (!element)
        return failure();

      result.push_back(element);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseGreater())
      return failure();

    return success();
  }

  static ParseResult parseTypeArguments(
    DialectAsmParser& parser, llvm::SmallVectorImpl<Type>& result)
  {
    if (parser.parseOptionalLSquare())
      return success();

    if (succeeded(parser.parseOptionalRSquare()))
      return success();

    do
    {
      mlir::Type element = parseVeronaType(parser);
      if (!element)
        return failure();

      result.push_back(element);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseOptionalRSquare())
      return failure();

    return success();
  }

  static Type parseMeetType(MLIRContext* ctx, DialectAsmParser& parser)
  {
    SmallVector<mlir::Type, 2> elements;
    if (parseTypeList(parser, elements))
      return Type();
    return MeetType::get(ctx, elements);
  }

  static Type parseJoinType(MLIRContext* ctx, DialectAsmParser& parser)
  {
    SmallVector<mlir::Type, 2> elements;
    if (parseTypeList(parser, elements))
      return Type();
    return JoinType::get(ctx, elements);
  }

  static Type parseIntegerType(
    MLIRContext* ctx, DialectAsmParser& parser, StringRef keyword)
  {
    size_t width = 0;
    if (keyword.substr(1).getAsInteger(10, width))
    {
      parser.emitError(parser.getNameLoc(), "unknown verona type: ") << keyword;
      return Type();
    }
    bool sign = keyword.startswith("S");
    return IntegerType::get(ctx, width, sign);
  }

  static Type parseClassType(MLIRContext* ctx, DialectAsmParser& parser)
  {
    FlatSymbolRefAttr name;
    SmallVector<Type, 4> arguments;
    if (
      parser.parseLess() || parser.parseAttribute(name) ||
      parseTypeArguments(parser, arguments) || parser.parseGreater())
      return Type();

    return ClassType::get(ctx, name.getValue(), arguments);
  }

  static Type parseViewpointType(MLIRContext* ctx, DialectAsmParser& parser)
  {
    Type left;
    Type right;
    if (
      parser.parseLess() || !(left = parseVeronaType(parser)) ||
      parser.parseComma() || !(right = parseVeronaType(parser)) ||
      parser.parseGreater())
      return Type();

    return ViewpointType::get(ctx, left, right);
  }

  static Type parseVariableType(MLIRContext* ctx, DialectAsmParser& parser)
  {
    uint64_t index;

    if (
      parser.parseLess() || parser.parseInteger(index) || parser.parseGreater())
      return Type();

    return VariableType::get(ctx, index);
  }

  Type parseVeronaType(DialectAsmParser& parser)
  {
    MLIRContext* ctx = parser.getBuilder().getContext();

    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return Type();

    if (keyword == "class")
      return parseClassType(ctx, parser);
    else if (keyword == "meet")
      return parseMeetType(ctx, parser);
    else if (keyword == "join")
      return parseJoinType(ctx, parser);
    else if (keyword == "viewpoint")
      return parseViewpointType(ctx, parser);
    else if (keyword == "top")
      return MeetType::get(ctx, {});
    else if (keyword == "bottom")
      return JoinType::get(ctx, {});
    else if (keyword == "iso")
      return CapabilityType::get(ctx, Capability::Isolated);
    else if (keyword == "mut")
      return CapabilityType::get(ctx, Capability::Mutable);
    else if (keyword == "imm")
      return CapabilityType::get(ctx, Capability::Immutable);
    else if (keyword == "variable")
      return parseVariableType(ctx, parser);
    else if (keyword.startswith("U") || keyword.startswith("S"))
      return parseIntegerType(ctx, parser, keyword);

    parser.emitError(parser.getNameLoc(), "unknown verona type: ") << keyword;
    return Type();
  }

  static void printTypeList(ArrayRef<Type> types, DialectAsmPrinter& os)
  {
    os << "<";
    llvm::interleaveComma(
      types, os, [&](auto element) { printVeronaType(element, os); });
    os << ">";
  }

  static void printTypeArguments(ArrayRef<Type> types, DialectAsmPrinter& os)
  {
    os << "[";
    llvm::interleaveComma(
      types, os, [&](auto element) { printVeronaType(element, os); });
    os << "]";
  }

  void printVeronaType(Type type, DialectAsmPrinter& os)
  {
    switch (type.getKind())
    {
      case VeronaTypes::Integer:
      {
        auto iTy = type.cast<IntegerType>();
        if (iTy.getSign())
        {
          os << "S";
        }
        else
        {
          os << "U";
        }
        os << iTy.getWidth();
        break;
      }

      case VeronaTypes::Meet:
      {
        auto meetType = type.cast<MeetType>();
        if (meetType.getElements().empty())
        {
          os << "top";
        }
        else
        {
          os << "meet";
          printTypeList(meetType.getElements(), os);
        }
        break;
      }

      case VeronaTypes::Join:
      {
        auto joinType = type.cast<JoinType>();
        if (joinType.getElements().empty())
        {
          os << "bottom";
        }
        else
        {
          os << "join";
          printTypeList(joinType.getElements(), os);
        }
        break;
      }

      case VeronaTypes::Capability:
      {
        auto capType = type.cast<CapabilityType>();
        switch (capType.getCapability())
        {
          case Capability::Isolated:
            os << "iso";
            break;
          case Capability::Mutable:
            os << "mut";
            break;
          case Capability::Immutable:
            os << "imm";
            break;
        }
        break;
      }

      case VeronaTypes::Class:
      {
        auto classType = type.cast<ClassType>();
        auto arguments = classType.getArguments();
        os << "class<@" << classType.getClassName();
        if (!arguments.empty())
        {
          printTypeArguments(arguments, os);
        }
        os << ">";
        break;
      }

      case VeronaTypes::Viewpoint:
      {
        auto viewpointType = type.cast<ViewpointType>();
        os << "viewpoint<" << viewpointType.getLeftType() << ", "
           << viewpointType.getRightType() << ">";
        break;
      }

      case VeronaTypes::Variable:
      {
        auto varType = type.cast<VariableType>();
        os << "variable<" << varType.getIndex() << ">";
        break;
      }
    }
  }

  bool isaVeronaType(Type type)
  {
    return type.getKind() >= FIRST_VERONA_TYPE &&
      type.getKind() < LAST_VERONA_TYPE;
  }

  bool areVeronaTypes(llvm::ArrayRef<Type> types)
  {
    return llvm::all_of(types, isaVeronaType);
  }

  /// Distribute a lattice type (join or meet) by applying `f` to every element
  /// of it. Each return value of the continuation is added to `result`.
  ///
  /// Assuming `type` is in normal form, this method will process nested `T`s
  /// as well.
  ///
  /// For example, given `join<A, join<B, C>>`, this method will add
  /// `f(A), f(B), f(C)` to `result`.
  template<typename T>
  static void distributeType(
    SmallVectorImpl<Type>& result, T type, llvm::function_ref<Type(Type)> f)
  {
    for (Type element : type.getElements())
    {
      if (auto nested = element.dyn_cast<T>())
        distributeType<T>(result, nested, f);
      else
        result.push_back(f(element));
    }
  }

  /// If the argument `type` is of kind `T` (where `T` is a lattice type, ie.
  /// JoinType or MeetType), distribute it by applying `f` to every element of
  /// it. The return values are combined to form a new lattice type of the same
  /// kind. If `type` is not of kind `T`, it is directly applied to `f`.
  ///
  /// Assuming `type` is in normal form, this method will process nested `T`s
  /// as well.
  ///
  /// For example, given `join<A, join<B, C>>`, this method will return
  /// `join<f(A), f(B), f(C)>`.
  template<typename T>
  static Type
  distributeType(MLIRContext* ctx, Type type, llvm::function_ref<Type(Type)> f)
  {
    if (auto node = type.dyn_cast<T>())
    {
      SmallVector<Type, 4> result;
      distributeType<T>(result, node, f);
      return T::get(ctx, result);
    }
    else
    {
      return f(type);
    }
  }

  /// Distribute all join and meets found in `type`, by applying `f` to every
  /// "atom" in the type. `type` is assumed to be in normal form already.
  ///
  /// For example, given `join<meet<A, B>, C>`, this function returns
  /// `join<meet<f(A), f(B)>, f(C)>`.
  static Type
  distributeAll(MLIRContext* ctx, Type type, llvm::function_ref<Type(Type)> f)
  {
    return distributeType<JoinType>(ctx, type, [&](Type inner) {
      return distributeType<MeetType>(ctx, inner, f);
    });
  }

  /// Normalize a meet type.
  /// This function returns the normal form of `meet<normalized..., rest...>`,
  /// distributing any nested joins.
  ///
  /// Types in `normalized` must be in normal form and not contain any joins.
  /// Types in `rest` may be in any form.
  ///
  /// This method uses `normalized` as scratch space; it recurses with more
  /// elements pushed to it. When it returns, `normalized` will always have its
  /// original length and contents.
  ///
  /// TODO: this function uses recursion to iterate over the `rest` array,
  /// because that works well with normalizeType. It could be rewritten to use
  /// loops, which is probably more efficient and doesn't risk blowing the
  /// stack.
  Type normalizeMeet(
    Operation* op,
    TypePolarity polarity,
    SmallVectorImpl<Type>& normalized,
    ArrayRef<Type> rest)
  {
    MLIRContext* ctx = op->getContext();

    if (rest.empty())
      return MeetType::get(ctx, normalized);

    Type element = normalizeType(op, polarity, rest.front());
    return distributeType<JoinType>(ctx, element, [&](auto inner) {
      normalized.push_back(inner);
      auto result = normalizeMeet(op, polarity, normalized, rest.drop_front());
      normalized.pop_back();
      return result;
    });
  }

  /// Normalize a meet type.
  /// This function returns the normal form of `meet<elements...>`,
  /// distributing any nested joins.
  Type
  normalizeMeet(Operation* op, TypePolarity polarity, ArrayRef<Type> elements)
  {
    SmallVector<Type, 4> result;
    return normalizeMeet(op, polarity, result, elements);
  }

  /// Normalize a join type.
  /// This function returns the normal form of `join<elements...>`. The only
  /// effect of this is individually normalizing the contents of `elements`.
  Type
  normalizeJoin(Operation* op, TypePolarity polarity, ArrayRef<Type> elements)
  {
    SmallVector<Type, 4> result;
    llvm::transform(elements, std::back_inserter(result), [&](Type element) {
      return normalizeType(op, polarity, element);
    });
    return JoinType::get(op->getContext(), result);
  }

  Type normalizeViewpoint(
    Operation* op, TypePolarity polarity, Type left, Type right)
  {
    Type normalizedLeft = normalizeType(op, polarity, left);
    Type normalizedRight = normalizeType(op, polarity, right);

    MLIRContext* ctx = op->getContext();
    return distributeAll(ctx, normalizedLeft, [&](Type distributedLeft) {
      return distributeAll(ctx, normalizedRight, [&](Type distributedRight) {
        return ViewpointType::get(ctx, distributedLeft, distributedRight);
      });
    });
  }

  Type normalizeType(Operation* op, TypePolarity polarity, Type type)
  {
    assert(isaVeronaType(type));
    switch (type.getKind())
    {
      // These don't contain any nested types and need no expansion.
      case VeronaTypes::Integer:
      case VeronaTypes::Capability:
        return type;

      // We don't normalize type parameters inside a Class type, as we don't
      // know which polarity to pick.
      case VeronaTypes::Class:
        return type;

      case VeronaTypes::Variable:
        switch (polarity)
        {
          case TypePolarity::Positive:
            return type;

          case TypePolarity::Negative:
          {
            SmallVector<Type, 4> normalized = {type};
            Type bound = PolymorphicOp::getTypeVariableBound(
              op, type.cast<VariableType>());
            return normalizeMeet(op, polarity, normalized, {bound});
          }
        }

      case VeronaTypes::Join:
        return normalizeJoin(op, polarity, type.cast<JoinType>().getElements());

      case VeronaTypes::Meet:
        return normalizeMeet(op, polarity, type.cast<MeetType>().getElements());

      case VeronaTypes::Viewpoint:
      {
        auto viewpoint = type.cast<ViewpointType>();
        return normalizeViewpoint(
          op, polarity, viewpoint.getLeftType(), viewpoint.getRightType());
      }

      default:
        abort();
    }
  }

  FieldOp lookupClassField(ClassOp classOp, StringRef name)
  {
    assert(classOp != nullptr);
    for (Operation& op : *classOp.getBody())
    {
      if (verona::FieldOp fieldOp = dyn_cast<verona::FieldOp>(op);
          fieldOp && fieldOp.name() == name)
      {
        return fieldOp;
      }
    }
    return nullptr;
  }

  bool lookupFieldTypes(
    Operation* op,
    ArrayRef<Type> origins,
    StringRef name,
    SmallVectorImpl<Type>& readElements,
    SmallVectorImpl<Type>& writeElements)
  {
    bool complete = true;
    for (Type origin : origins)
    {
      auto [readType, writeType] = lookupFieldType(op, origin, name);
      assert((readType == nullptr) == (writeType == nullptr));
      if (readType != nullptr)
      {
        readElements.push_back(readType);
        writeElements.push_back(writeType);
      }
      else
      {
        complete = false;
      }
    }

    return complete;
  }

  std::pair<Type, Type>
  lookupFieldType(Operation* op, Type origin, StringRef name)
  {
    MLIRContext* ctx = op->getContext();
    switch (origin.getKind())
    {
      case VeronaTypes::Meet:
      {
        auto meetType = origin.cast<MeetType>();
        SmallVector<Type, 4> readElements;
        SmallVector<Type, 4> writeElements;
        lookupFieldTypes(
          op, meetType.getElements(), name, readElements, writeElements);

        assert(readElements.size() == writeElements.size());
        switch (readElements.size())
        {
          case 0:
            return {nullptr, nullptr};
          case 1:
            return {readElements.front(), writeElements.front()};
          default:
            return {MeetType::get(ctx, readElements),
                    JoinType::get(ctx, writeElements)};
        }
      }

      case VeronaTypes::Join:
      {
        auto joinType = origin.cast<JoinType>();
        SmallVector<Type, 4> readElements;
        SmallVector<Type, 4> writeElements;

        bool complete = lookupFieldTypes(
          op, joinType.getElements(), name, readElements, writeElements);

        if (!complete)
          return {nullptr, nullptr};

        assert(readElements.size() == writeElements.size());
        if (readElements.size() == 1)
          return {readElements.front(), writeElements.front()};
        else
          return {JoinType::get(ctx, readElements),
                  MeetType::get(ctx, writeElements)};
      }

      case VeronaTypes::Class:
      {
        auto classType = origin.cast<ClassType>();
        ClassOp classOp = SymbolTable::lookupNearestSymbolFrom<ClassOp>(
          op, classType.getClassName());
        FieldOp fieldOp = lookupClassField(classOp, name);
        if (!fieldOp)
          return {nullptr, nullptr};

        Type fieldType = PolymorphicOp::replaceTypeVariables(
          fieldOp.type(), classType.getArguments());
        return {fieldType, fieldType};
      }

      case VeronaTypes::Viewpoint:
        return lookupFieldType(
          op, origin.cast<ViewpointType>().getRightType(), name);

      default:
        return {nullptr, nullptr};
    }
  }

  uint64_t typeVariablesInScope(Operation* op)
  {
    uint64_t count = 0;
    for (; op != nullptr; op = op->getParentOp())
    {
      if (op->hasTrait<OpTrait::PolymorphicOpTrait>())
        count += PolymorphicOp::getNumTypeParameters(op);
    }
    return count;
  }

  uint64_t freeTypeVariables(Type type)
  {
    switch (type.getKind())
    {
      case VeronaTypes::Integer:
        return 0;

      case VeronaTypes::Meet:
      {
        auto meetType = type.cast<MeetType>();
        uint64_t result = 0;
        for (auto it : meetType.getElements())
          result = std::max(freeTypeVariables(it), result);
        return result;
      }

      case VeronaTypes::Join:
      {
        auto joinType = type.cast<JoinType>();
        uint64_t result = 0;
        for (auto it : joinType.getElements())
          result = std::max(freeTypeVariables(it), result);
        return result;
      }

      case VeronaTypes::Variable:
      {
        auto varType = type.cast<VariableType>();
        return varType.getIndex() + 1;
      }

      case VeronaTypes::Capability:
      case VeronaTypes::Class:
        return 0;

      default:
        abort();
    }
  }
}
