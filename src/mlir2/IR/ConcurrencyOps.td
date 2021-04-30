#ifndef VERONA_IR_CONCURRENCY_OPS_TD
#define VERONA_IR_CONCURRENCY_OPS_TD

def VAM_WhenOp : VAM_Op<"when", [AttrSizedOperandSegments, IsolatedFromAbove]> {
  let arguments = (ins Variadic<VAM_Value>:$cowns, Variadic<VAM_Value>:$captures);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    ( `cowns` `=` `(` $cowns^ `)` )?
    ( `captures` `=` `(` $captures^ `)` )?
    `behaviour` `=` custom<VeronaBody>($body)
    attr-dict
  }];
}

def VAM_NewCownOp : VAM_Op<"new_cown", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$descriptor, VAM_Value:$operand);
  let results = (outs VAM_Value:$result);
  let assemblyFormat = [{ $descriptor `,` $operand attr-dict }];
}

#endif
