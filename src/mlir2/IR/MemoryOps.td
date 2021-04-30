#ifndef VERONA_IR_MEMORY_OPS_TD
#define VERONA_IR_MEMORY_OPS_TD

include "mlir/IR/SymbolInterfaces.td"

def VAM_FreezeOp : VAM_UnaryOp<"freeze"> { }
def VAM_MergeOp : VAM_UnaryOp<"merge"> { }

def VAM_TraceRegionOp : VAM_Op<"trace_region"> {
  let arguments = (ins VAM_Value:$operand);
  let assemblyFormat = [{ $operand attr-dict }];
}

def VAM_NewRegionOp : VAM_Op<"new_region", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$descriptor);
  let results = (outs VAM_Value:$result);
  let assemblyFormat = [{ $descriptor attr-dict }];
}

def VAM_NewObjectOp : VAM_Op<"new_object", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$descriptor, VAM_Value:$parent);
  let results = (outs VAM_Value:$result);
  let assemblyFormat = [{ $descriptor `in` $parent attr-dict }];
}

def VAM_LoadDescriptorOp : VAM_Op<"load_descriptor", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$descriptor);
  let results = (outs VAM_Value:$result);
  let assemblyFormat = [{ $descriptor attr-dict }];
}

def VAM_LoadFieldOp : VAM_Op<"load_field"> {
  let arguments = (ins VAM_Value:$origin, SelectorAttr:$selector);
  let results = (outs VAM_Value:$result);
  let assemblyFormat = [{ $origin `:` $selector attr-dict }];
}

def VAM_StoreFieldOp : VAM_Op<"store_field"> {
  let arguments = (ins VAM_Value:$origin, SelectorAttr:$selector, VAM_Value:$value);
  let results = (outs VAM_Value:$result);
  let assemblyFormat = [{ $origin `:` $selector `,` $value attr-dict }];
}

def VAM_ProtectOp : VAM_Op<"protect"> {
  let arguments = (ins Variadic<VAM_Value>:$operands);
  let assemblyFormat = [{ $operands attr-dict }];
}

def VAM_UnprotectOp : VAM_Op<"unprotect"> {
  let arguments = (ins Variadic<VAM_Value>:$operands);
  let assemblyFormat = [{ $operands attr-dict }];
}

#endif
