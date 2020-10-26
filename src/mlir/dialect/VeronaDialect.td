// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#ifndef VERONA_DIALECT
#define VERONA_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Verona dialect definition.
//===----------------------------------------------------------------------===//

def Verona_Dialect : Dialect {
    let name = "verona";
    let summary = "A prototype Verona language MLIR dialect.";
    let cppNamespace = "::mlir::verona";
}

//===----------------------------------------------------------------------===//
// Base verona operation/type definitions.
//===----------------------------------------------------------------------===//

class Verona_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Verona_Dialect, mnemonic, traits> {
    let parser = [{ return ::mlir::verona::parse$cppClass(parser, result); }];
    let printer = [{ return ::mlir::verona::print(p, *this); }];
    let verifier = [{ return ::mlir::verona::verify(*this); }];
}

class Verona_TypeDef<string name> : TypeDef<Verona_Dialect, name> { }

def Verona_Type : DialectType<Verona_Dialect,
    CPred<"isaVeronaType($_self)">, "Verona type">;

def IsMetaType : DialectType<Verona_Dialect,
  CPred<"$_self.isa<::mlir::verona::MetaType>()">, "test">,
  BuildableType<"$_builder.getType<::mlir::verona::MetaType>()">;

def IsValueType : DialectType<Verona_Dialect,
  CPred<"$_self.isa<::mlir::verona::ValueType>()">, "test">,
  BuildableType<"$_builder.getType<::mlir::verona::ValueType>()">;

#endif // VERONA_DIALECT
