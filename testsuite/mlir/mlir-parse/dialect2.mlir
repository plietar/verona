// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is a simple example to demonstrate the Verona MLIR dialect. The types
// used are just a placeholder and not correct yet.
//

module {
  verona.class @Cell[!verona.top] {
    verona.field "contents" : !verona.variable<0>
  }

  verona.func @Cell_get[!verona.top, !verona.join<mut, imm>](%cell: !verona.meet<class<@Cell[variable<0>]>, variable<1>>) -> !verona.viewpoint<variable<1>, variable<0>> {
    %contents = verona.field_read %cell["contents"] : !verona.meet<class<@Cell[variable<0>]>, variable<1>> -> !verona.viewpoint<variable<1>, variable<0>>
    verona.return %contents : !verona.viewpoint<variable<1>, variable<0>>
  }

  // %cell: @Cell[$0] & mut, %value: $0

  verona.func @Cell_extract[!verona.top](%cell: !verona.meet<class<@Cell[variable<0>]>, mut>, %value: !verona.variable<0>) -> !verona.variable<0> {
    %contents = verona.field_write %cell["contents"], %value : !verona.meet<class<@Cell[variable<0>]>, mut> -> !verona.variable<0> -> !verona.variable<0>
    verona.return %contents : !verona.variable<0>
  }

  // verona.func @Cell_get_bad[!verona.top, !verona.join<mut, imm>](%cell: !verona.meet<class<@Cell[variable<0>]>, variable<1>>) -> !verona.variable<0> {
  //   %contents = verona.field_read %cell["contents"] : !verona.meet<class<@Cell[variable<0>]>, variable<1>> -> !verona.variable<0>
  //   verona.return %contents : !verona.variable<0>
  // }
}

