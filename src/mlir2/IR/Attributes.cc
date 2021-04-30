#include "mlir2/IR/Attributes.h"

#include "mlir/IR/Builders.h"

namespace verona::vam
{
  using bytecode::SelectorIdx;
  namespace detail
  {
    struct SelectorAttributeStorage : public mlir::AttributeStorage
    {
      using KeyTy = SelectorIdx::underlying_type;

      bool operator==(const KeyTy& key) const
      {
        return key == value.value;
      }

      static SelectorAttributeStorage*
      construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
      {
        return new (allocator.allocate<SelectorAttributeStorage>())
          SelectorAttributeStorage(SelectorIdx(key));
      }

      SelectorAttributeStorage(bytecode::SelectorIdx value) : value(value) {}
      bytecode::SelectorIdx value;
    };
  }

  SelectorAttr
  SelectorAttr::get(bytecode::SelectorIdx value, mlir::MLIRContext* context)
  {
    return Base::get(context, value.value);
  }

  bytecode::SelectorIdx SelectorAttr::getValue() const
  {
    return getImpl()->value;
  }

  void SelectorAttr::print(mlir::DialectAsmPrinter& os) const
  {
    os << "selector<" << getValue().value << ">";
  }

  SelectorAttr SelectorAttr::parse(mlir::DialectAsmParser& parser)
  {
    uint32_t value;
    if (parser.parseLess())
      return nullptr;
    if (parser.parseInteger(value))
      return nullptr;
    if (parser.parseGreater())
      return nullptr;

    return SelectorAttr::get(
      SelectorIdx(value), parser.getBuilder().getContext());
  }
}
