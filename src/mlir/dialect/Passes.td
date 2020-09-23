// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#ifndef VERONA_PASSES
#define VERONA_PASSES

include "mlir/Pass/PassBase.td"

def Typechecker : Pass<"typechecker"> {
  let summary = "Run the Verona typechecker on the IR.";
  let constructor = "createTypecheckerPass()";
}

#endif // VERONA_PASSES
