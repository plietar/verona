#ifndef VERONA_IR_MISC_OPS_TD
#define VERONA_IR_MISC_OPS_TD

def VAM_CopyOp : VAM_UnaryOp<"copy"> { }
def VAM_MutViewOp : VAM_UnaryOp<"mut_view"> { }

def VAM_DropOp : VAM_Op<"drop"> {
  let arguments = (ins VAM_Value:$operand);
  let assemblyFormat = [{ $operand attr-dict }];
}

def VAM_PrintOp : VAM_Op<"print"> {
  let arguments = (ins VAM_Value:$format, Variadic<VAM_Value>:$values);
  let assemblyFormat = [{ $format (`,` $values^)? attr-dict }];
}

def VAM_CallOp : VAM_Op<"call"> {
  let arguments = (ins VAM_Value:$receiver, SelectorAttr:$selector, Variadic<VAM_Value>:$args);
  let results = (outs VAM_Value:$result);
  let assemblyFormat = [{ $selector `(` $receiver (`,` $args^)? `)` attr-dict }];
}

def VAM_LiteralOp : VAM_Op<"literal"> {
  let arguments = (ins AnyAttrOf<[StrAttr, I64Attr, UnitAttr]>:$value);
  let results = (outs VAM_Value:$result);
  let assemblyFormat = [{ $value attr-dict }];
}

def VAM_MatchDescriptorOp : VAM_Op<"match_descriptor"> {
  let arguments = (ins VAM_Value:$operand, SymbolRefAttr:$descriptor);
  let results = (outs I1:$result);
}

// def VAM_MatchCapabilityOp : VAM_BinOp<"match_capability"> { }

def VAM_ReturnOp : VAM_Op<"return", [Terminator, ParentOneOf<["::mlir::FuncOp", "WhenOp"]>]> {
  let arguments = (ins VAM_Value:$operand);
  let assemblyFormat = [{ $operand attr-dict }];
}

def VAM_TruthinessOp : VAM_Op<"truthiness"> {
  let arguments = (ins VAM_Value:$operand);
  let results = (outs I1:$result);
  let assemblyFormat = [{ $operand attr-dict }];
}

#endif
