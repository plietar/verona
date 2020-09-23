// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "ast/ast.h"
#include "dialect/VeronaDialect.h"
#include "error.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/SourceMgr.h"

namespace mlir::verona
{
  /**
   * Main compiler API.
   *
   * The driver is user by first calling one of the `readXXX` methods, followed
   * by `emitMLIR`. The various `readXXX` methods allow using different kinds of
   * input.
   *
   * The lowering pipeline is configured through Driver's constructor arguments.
   *
   * For now, the error handling is crude and needs proper consideration,
   * especially aggregating all errors and context before sending it back to
   * the public API callers.
   */
  class Driver
  {
  public:
    Driver(
      const PassPipelineCLParser& pipelineParser,
      unsigned optLevel = 0,
      bool enableDiagnosticsVerifier = false);

    // TODO: add a readSource function that parses Verona source code.
    // this might be more thinking about the error API of the Driver.

    /// Lower an AST into an MLIR module, which is loaded in the driver.
    llvm::Error readAST(const ::ast::Ast& ast);

    /// Read textual MLIR into the driver's module.
    llvm::Error readMLIR(std::unique_ptr<llvm::MemoryBuffer> buffer);

    /// Emit the module as textual MLIR.
    llvm::Error emitMLIR(llvm::raw_ostream& os);

    /// Check that diagnostics emitted by the MLIR pipeline match the
    /// expectations set in the source.
    ///
    /// The Driver must have been initialized with
    /// enableDiagnosticsVerifier = true.
    llvm::Error verifyDiagnostics();

    /// Dump the current IR to an ostream.
    ///
    /// Unlike emitMLIR, this may be used even in failure conditions, as
    /// debugging information. If the driver holds no module (eg. lowering from
    /// AST failed), nothing happens.
    void dumpMLIR(llvm::raw_ostream& os);

  private:
    /// MLIR context.
    mlir::MLIRContext context;

    /// MLIR module.
    /// It gets modified as the driver progresses through its passes.
    mlir::OwningModuleRef module;

    /// MLIR Pass Manager
    /// It gets configured by the constructor based on the provided arguments.
    mlir::PassManager passManager;

    /// Source manager.
    llvm::SourceMgr sourceManager;

    /// Diagnostic handler that pretty-prints MLIR errors.
    ///
    /// The handler registers itself with the MLIR context and gets invoked
    /// automatically. We only need to keep it alive by storing it here.
    ///
    /// If the Driver was initialized with enableDiagnosticsVerifier = true,
    /// this will be a SourceMgrDiagnosticVerifierHandler.
    std::unique_ptr<SourceMgrDiagnosticHandler> diagnosticHandler;

    bool enableDiagnosticsVerifier;
  };
}
