#include "mlir/IR/OpImplementation.h"
#include "mlir2/IR/Dialect.h"

namespace verona::vam
{
  mlir::ParseResult parseVeronaBody(
    mlir::OpAsmParser& parser,
    mlir::Region& region,
    std::optional<std::reference_wrapper<mlir::NamedAttrList>> attrs =
      std::nullopt);

  void printVeronaBody(
    mlir::OpAsmPrinter& printer,
    mlir::Region& region,
    mlir::DictionaryAttr attrs = nullptr,
    llvm::ArrayRef<llvm::StringRef> elidedAttrs = {});

  inline void printVeronaBody(
    mlir::OpAsmPrinter& printer,
    MethodOp op,
    mlir::Region& region,
    mlir::DictionaryAttr attrs)
  {
    printVeronaBody(printer, region, attrs, {"selector", "name"});
  }
  inline void
  printVeronaBody(mlir::OpAsmPrinter& printer, WhenOp op, mlir::Region& region)
  {
    printVeronaBody(printer, region);
  }

  /*
  mlir::ParseResult
  parseMethodOp(mlir::OpAsmParser& parser, mlir::OperationState& state);
  void printOp(mlir::OpAsmPrinter& printer, MethodOp op);
  */
}
