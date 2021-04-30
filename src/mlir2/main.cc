#include "compiler/elaboration.h"
#include "compiler/parser.h"
#include "compiler/resolution.h"
#include "compiler/typecheck/wf_types.h"
#include "interpreter/interpreter.h"
#include "interpreter/options.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir2/IR/Dialect.h"
#include "mlir2/generator.h"
#include "mlir2/lower.h"

#include "llvm/Support/SourceMgr.h"

using namespace verona;

struct Options : public interpreter::InterpreterOptions
{
  std::string input;
  std::optional<std::string> output_file;

  void configure(CLI::App& app)
  {
    verona::interpreter::add_arguments(app, *this, "run");
    app.add_option("input", input, "Input file")->required();
    app.add_option("--output", output_file, "Output file");
  }

  void validate()
  {
    verona::interpreter::validate_args(*this);
  }
};

namespace verona::vam
{
  struct AtFunctionExit
  {
    std::function<void()> closure;
    AtFunctionExit(std::function<void()> closure) : closure(closure) {}
    ~AtFunctionExit()
    {
      closure();
    };
  };

  mlir::OwningModuleRef
  load_source(mlir::MLIRContext& mlir_context, std::string_view path)
  {
    compiler::Context context;
    AtFunctionExit print_diagnostic(
      [&]() { return context.print_diagnostic_summary(); });

    auto program = std::make_unique<compiler::Program>();

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open())
    {
      std::cerr << "Cannot open file \"" << path << "\"" << std::endl;
      return nullptr;
    }

    context.add_source_file(std::string(path));
    auto file = compiler::parse(context, std::string(path), input);
    if (!file)
    {
      std::cerr << "Parsing failed" << std::endl;
      return nullptr;
    }
    program->files.push_back(std::move(file));

    if (!compiler::name_resolution(context, program.get()))
      return nullptr;
    if (!compiler::elaborate(context, program.get()))
      return nullptr;
    if (!compiler::check_wf_types(context, program.get()))
      return nullptr;
    auto analysis = compiler::analyse(context, program.get());
    if (!analysis->ok)
      return nullptr;

    return lower(&mlir_context, context, *program, *analysis);
  }

  mlir::OwningModuleRef
  load_mlir(mlir::MLIRContext& context, std::string_view path)
  {
    auto file = llvm::MemoryBuffer::getFileOrSTDIN(llvm::StringRef(path));
    if (std::error_code ec = file.getError())
    {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
      return nullptr;
    }

    llvm::SourceMgr sourceManager;
    sourceManager.AddNewSourceBuffer(std::move(*file), llvm::SMLoc());
    return mlir::parseSourceFile(sourceManager, &context);
  }
}

void execute(Options& options)
{
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<vam::VeronaAbstractMachine>();

  mlir::OwningModuleRef module;
  if (llvm::StringRef(options.input).endswith(".mlir"))
  {
    module = verona::vam::load_mlir(context, options.input);
  }
  else
  {
    module = verona::vam::load_source(context, options.input);
  }

  if (!module)
  {
    exit(1);
  }

  if (failed(mlir::verify(*module)))
  {
    std::cerr << "Invalid module" << std::endl;
    module->print(llvm::errs());
    exit(1);
  }

  std::vector<uint8_t> bytecode;
  if (options.output_file || options.run)
  {
    bytecode = verona::vam::generate(*module);
  }

  if (options.output_file)
  {
    std::ofstream output(*options.output_file, std::ios::binary);
    if (!output.is_open())
    {
      std::cerr << "Cannot open file " << *options.output_file << std::endl;
      exit(1);
    }

    output.write(
      reinterpret_cast<const char*>(bytecode.data()), bytecode.size());
  }

  if (options.run)
  {
    interpreter::Code code(bytecode);
    interpreter::instantiate(options, code);
  }
  else
  {
    module->print(llvm::outs());
  }
}

int main(int argc, const char** argv)
{
  // llvm::InitLLVM y(argc, argv);

  CLI::App app{"Verona compiler"};

  Options options;
  options.configure(app);
  CLI11_PARSE(app, argc, argv);
  options.validate();

  execute(options);
}
