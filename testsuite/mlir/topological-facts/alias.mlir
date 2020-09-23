// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

// Value and block names in the compiler output (as used in the CHECK
// directives) is unfortunately not guaranteed by the compiler to match the
// names found in source file. As a workaround, we intentionally use the same
// names the compiler would have used in the source.

module {
  func @test_alias_reflexive(%arg0: !verona.mut) {
    // CHECK-L: @test_alias_reflexive:
    // CHECK-L:   alias(%arg0, %arg0)
    return
  }

  func @test_alias_copy(%arg0: !verona.mut) {
    // CHECK-L: @test_alias_copy:
    // CHECK-L:   alias(%0, %arg0)
    // CHECK-L:   alias(%arg0, %0)
    %0 = verona.copy %arg0 : !verona.mut -> !verona.mut
    return
  }

  func @test_alias_view(%arg0: !verona.mut) {
    // CHECK-L: @test_alias_view:
    // CHECK-L:   alias(%0, %arg0)
    // CHECK-L:   alias(%arg0, %0)
    %0 = verona.view %arg0 : !verona.mut -> !verona.mut
    return
  }

  func @test_alias_transitive(%arg0: !verona.mut) {
    // CHECK-L: @test_alias_transitive:
    // CHECK-L:   alias(%1, %arg0)
    // CHECK-L:   alias(%arg0, %1)
    %0 = verona.view %arg0 : !verona.mut -> !verona.mut
    %1 = verona.view %0 : !verona.mut -> !verona.mut
    return
  }

  // Because the alias fact is reflexive, alias(%arg0, %arg0) holds in the entry
  // block. After block renamings are applied, this fact occurs with all
  // combinations of new names %arg0 has (ie. %arg0, %0 and %1).
  func @test_alias_block_argument(%arg0: !verona.mut) {
    // CHECK-L: @test_alias_block_argument:
    // CHECK-L:   ^bb0:
    // CHECK-L:     alias(%arg0, %arg0)
    //
    // CHECK-L:   ^bb1:
    // CHECK-L:     alias(%0, %1)
    // CHECK-L:     alias(%0, %arg0)
    // CHECK-L:     alias(%1, %0)
    // CHECK-L:     alias(%1, %arg0)
    // CHECK-L:     alias(%arg0, %0)
    // CHECK-L:     alias(%arg0, %1)
    br ^bb1(%arg0, %arg0: !verona.mut, !verona.mut)

  ^bb1(%0: !verona.mut, %1: !verona.mut):
    return
  }

  // Variable %2 is either a copy or a view of %arg1, depending on the value of
  // %arg0. In either case, the intermediary value (%0 or %1) is an alias of
  // %arg1. Therefore after the join in bb3, the fact alias(%2, %arg1) holds.
  func @test_alias_intersect(%arg0 : i1, %arg1: !verona.mut) {
    // CHECK-L: @test_alias_intersect:
    // CHECK-L:   ^bb0:
    // CHECK-L:     alias(%arg1, %arg1)
    // CHECK-L:   ^bb1:
    // CHECK-L:     alias(%0, %arg1)
    // CHECK-L:   ^bb2:
    // CHECK-L:     alias(%1, %arg1)
    // CHECK-L:   ^bb3:
    // CHECK-L:     alias(%2, %arg1)

    cond_br %arg0, ^bb1, ^bb2
 
  ^bb1:
    %0 = verona.copy %arg1 : !verona.mut -> !verona.mut
    br ^end(%0 : !verona.mut)

  ^bb2:
    %1 = verona.view %arg1 : !verona.mut -> !verona.mut
    br ^end(%1 : !verona.mut)

  ^end(%2: !verona.mut):
    return
  }
}
