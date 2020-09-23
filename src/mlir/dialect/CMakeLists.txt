# This mimicks add_mlir_dialect and add_mlir_interface's functionality,
# but for pass declarations.
function(add_mlir_passes name file)
  set(LLVM_TARGET_DEFINITIONS ${file}.td)
  mlir_tablegen(${file}.h.inc -gen-pass-decls -name ${name})
  add_public_tablegen_target(MLIR${name}PassesIncGen)
endfunction()

add_mlir_dialect(VeronaOps verona)
add_mlir_interface(TypecheckInterface)
add_mlir_passes(Verona Passes)

add_mlir_doc(VeronaDialect -gen-dialect-doc VeronaDialect Verona/)
add_mlir_doc(VeronaOps -gen-op-doc VeronaOps Verona/)

add_mlir_dialect_library(MLIRVerona
    VeronaDialect.cc
    VeronaOps.cc
    VeronaTypes.cc
    Typechecker.cc
    TypecheckInterface.cc
    TypeSyntax.cc

    DEPENDS
    MLIRVeronaOpsIncGen
    MLIRTypecheckInterfaceIncGen
    MLIRVeronaPassesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
