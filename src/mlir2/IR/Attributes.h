#pragma once

#include "bytecode/bytecode.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

namespace verona::vam
{
  namespace detail
  {
    struct SelectorAttributeStorage;
  }

  class SelectorAttr
  : public mlir::Attribute::
      AttrBase<SelectorAttr, mlir::Attribute, detail::SelectorAttributeStorage>
  {
  public:
    using Base::Base;
    using ValueType = bytecode::SelectorIdx;

    static SelectorAttr
    get(bytecode::SelectorIdx value, mlir::MLIRContext* context);

    bytecode::SelectorIdx getValue() const;

    void print(mlir::DialectAsmPrinter& os) const;
    static SelectorAttr parse(mlir::DialectAsmParser& parser);
  };
}
