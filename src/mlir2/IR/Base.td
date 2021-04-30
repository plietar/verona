#ifndef VERONA_IR_BASE_TD
#define VERONA_IR_BASE_TD

include "mlir/IR/OpBase.td"

def VeronaAbstractMachine : Dialect {
    let name = "vam";
    let summary = "A prototype Verona language MLIR dialect.";
    let cppNamespace = "::verona::vam";
}

def VAM_Value : TypeDef<VeronaAbstractMachine, "Value"> {
  let mnemonic = "value";
  let builderCall = "$_builder.getType<::verona::vam::ValueType>()";
}

def SelectorAttr : DialectAttr<VeronaAbstractMachine, CPred<"$_self.isa<::verona::vam::SelectorAttr>()">, "Selector index"> {
  let storageType = [{ ::verona::vam::SelectorAttr }];
  let returnType = [{ ::verona::bytecode::SelectorIdx }];
  let constBuilderCall = "::verona::vam::SelectorAttr::get($0, $_builder.getContext())";
}

class VAM_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<VeronaAbstractMachine, mnemonic, traits>
{
    // let assemblyFormat = [{ operands attr-dict }];
    // let parser = [{ return ::parse$cppClass(parser, result); }];
    // let printer = [{ return ::print(p, *this); }];
    // let verifier = [{ return ::verify(*this); }];
}

class VAM_UnaryOp<string mnemonic, list<OpTrait> traits = []>
  : VAM_Op<mnemonic, traits> {
  let arguments = (ins VAM_Value:$operand);
  let results = (outs VAM_Value:$result);
  let assemblyFormat = [{ operands attr-dict }];
}

class VAM_BinaryOp<string mnemonic, list<OpTrait> traits = []>
  : VAM_Op<mnemonic, traits> {
  let arguments = (ins VAM_Value:$operand1, VAM_Value:$operand2);
  let results = (outs VAM_Value:$result);
  let assemblyFormat = [{ operands attr-dict }];
}

class AnyAttrOf<list<Attr> allowedTypes, string summary = ""> : Attr<
  Or<!foreach(allowedtype, allowedTypes, allowedtype.predicate)>,
  !interleave(!foreach(t, allowedTypes, t.summary), " or ")>
{
  let storageType = "::mlir::Attribute";
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

#endif
