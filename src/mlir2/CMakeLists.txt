set(MLIR_DIR ${VERONA_LLVM_LOCATION}/lib/cmake/mlir)
find_package(MLIR REQUIRED CONFIG)
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS IR/Dialect.td)
mlir_tablegen(IR/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(IR/Ops.h.inc -gen-op-decls)
mlir_tablegen(IR/Ops.cpp.inc -gen-op-defs)
mlir_tablegen(IR/TypeDefs.h.inc -gen-typedef-decls)
mlir_tablegen(IR/TypeDefs.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRVerona2IncGen)

add_library(MLIRVerona2
  IR/Dialect.cc
  IR/OpSyntax.cc
  IR/Attributes.cc
)
add_dependencies(MLIRVerona2 MLIRVerona2IncGen)
target_link_libraries(MLIRVerona2 PUBLIC bytecode MLIRIR)

add_executable(verona-mlir2 main.cc)
target_link_libraries(verona-mlir2 PRIVATE CLI11::CLI11 interpreter veronac-lib)
target_link_libraries(verona-mlir2 PRIVATE
  MLIRVerona2
  MLIRParser
  MLIRStandard)
install(TARGETS verona-mlir2 RUNTIME DESTINATION .)
