// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is a simple example to demonstrate the Verona MLIR dialect. The types
// used are just a placeholder and not correct yet.
//
// ```verona
// class C {  }
// class D {
//   f: U64;
//   g: S32;
// }
//
// bar() {
//     let a = new C;
//     let b = view a;
//     let c = new D { f: b, g: b } in a;
//     c.g = c.f;
//
//     tidy(a);
//     drop(a);
// }
// ```

module {
  verona.class @C attributes { type_parameters = 0 } {
  }

  verona.class @D attributes { type_parameters = 1 } {
    verona.field "f" : !verona.variable<0>
  }

  func @bar() {
    %a = verona.new_region @C [ ] : !verona.U64

    verona.return
  }
}
