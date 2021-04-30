#ifndef VERONA_IR_STRUCTURE_OPS_TD
#define VERONA_IR_STRUCTURE_OPS_TD

include "mlir/IR/SymbolInterfaces.td"

def VAM_DescriptorOp : VAM_Op<"descriptor", [
  IsolatedFromAbove, NoRegionArguments, Symbol, SingleBlockImplicitTerminator<"DescriptorTerminatorOp">
]> {
  let arguments = (ins SymbolNameAttr:$sym_name, OptionalAttr<SelectorAttr>:$finaliser);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{ $sym_name attr-dict-with-keyword $body }];

  let builders = [OpBuilderDAG<(ins "::llvm::StringRef":$name)>];
  let extraClassDeclaration = [{
    static DescriptorOp create(::mlir::Location loc, ::llvm::StringRef name);
    mlir::LogicalResult verifyOp();
  }];
  let verifier = [{ return this->verifyOp(); }];
}

def DescriptorTerminatorOp : VAM_Op<"descriptor_end", [
  Terminator, HasParent<"DescriptorOp">
]> { }

def VAM_FieldOp : VAM_Op<"field", [
  IsolatedFromAbove, NoRegionArguments, HasParent<"DescriptorOp">
]> { 
  let arguments = (ins SelectorAttr:$selector);
  let assemblyFormat = [{ $selector attr-dict }];
}

def VAM_MethodOp : VAM_Op<"method", [
  IsolatedFromAbove, HasParent<"DescriptorOp">
]> {
  let arguments = (ins SelectorAttr:$selector, StrAttr:$name);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    static MethodOp create(::mlir::Location loc, bytecode::SelectorIdx selector, ::llvm::StringRef name);
  }];

  let assemblyFormat = [{
    $selector `name` `=` $name custom<VeronaBody>($body, attr-dict)
  }];
}

#endif
