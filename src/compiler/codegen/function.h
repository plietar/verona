// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "bytecode/generator.h"
#include "compiler/analysis.h"
#include "compiler/codegen/descriptor.h"

namespace verona::compiler
{
  using bytecode::CalleeRegister;
  using bytecode::Generator;
  using bytecode::Register;

  void emit_functions(
    Context& context,
    const AnalysisResults& analysis,
    const Reachability& reachability,
    ProgramTable& program_table,
    const SelectorTable& selectors,
    Generator& gen);

  struct FunctionABI
  {
    explicit FunctionABI(const FnSignature& sig)
    : arguments(1 + sig.parameters.size()), returns(1)
    {}

    explicit FunctionABI(const CallStmt& stmt)
    : arguments(1 + stmt.arguments.size()), returns(1)
    {}

    // Adds one to arguments for unused receiver
    // TODO-Better-Static-codegen
    // No output for now TODO-PROMISE
    explicit FunctionABI(const WhenStmt& stmt)
    : arguments(stmt.cowns.size() + stmt.captures.size() + 1), returns(1)
    {}

    /**
     * Number of arguments this function has.
     *
     * Note that the ABI always has a receiver argument, even if the function is
     * a static method. There may therefore be 1 register argument more than the
     * function has parameters.
     */
    size_t arguments;

    /**
     * Number of return values this function has, currently always 1.
     */
    size_t returns;

    /**
     * Get the space needed to pass arguments and return values.
     *
     * This is the size of the overlap between the caller's and the callee's
     * frames
     */
    size_t callspace() const
    {
      return std::max(arguments, returns);
    }

    static FunctionABI create_closure_abi(size_t count)
    {
      return FunctionABI(count);
    }

  private:
    // Adds one to arguments for unused receiver
    // TODO-Better-Static-codegen
    explicit FunctionABI(size_t count) : arguments(count + 1), returns(1) {}
  };

  class RegisterAllocator
  {
  public:
    RegisterAllocator(const FunctionABI& abi);

    /**
     * Allocate a new register.
     */
    Register get();

    /**
     * Reserve space at the top of the frame used to pass arguments during
     * function calls.
     *
     * The same space is used for all calls in the current function. Registers
     * returned by the `get` function will never overlap with this area.
     */
    void reserve_child_callspace(const FunctionABI& abi);

    /**
     * Get the total number of registers needed by the function.
     */
    uint8_t frame_size() const;

  private:
    size_t next_register_;
    size_t children_call_space_ = 0;
  };

  /**
   * Utility class for generating functions.
   *
   * This class provides emits the right header to the function, including
   * taking care of computing the function's body and frame size.
   */
  class BaseFunctionGenerator
  {
  public:
    /**
     * Initialize a function generator. Calling this constructor will
     * immediately write the function header to the bytecode.
     */
    BaseFunctionGenerator(
      Generator& gen, std::string_view name, FunctionABI abi)
    : gen_(gen),
      allocator_(abi),
      function_end_(gen.create_label()),
      frame_size_(gen.create_relocatable())
    {
      gen.str(name);
      gen.u8(truncate<uint8_t>(abi.arguments));
      gen.u8(truncate<uint8_t>(abi.returns));
      gen.u8(frame_size_);

      // We emit the size of the method using the function_end label, relative
      // to the end of the header. The header_end includes 4 bytes to account
      // for the size field itself.
      size_t header_end = gen.current_offset() + 4;
      gen.u32(function_end_, header_end);
      assert(gen.current_offset() == header_end);
    }

    /**
     * Emit an instruction to the bytecode.
     */
    template<bytecode::Opcode Op, typename... Ts>
    void emit(Ts&&... ts)
    {
      gen_.emit<Op>(std::forward<Ts>(ts)...);
    }

    /**
     * Finish generating the function.
     *
     * This must only be called once, after the function's body was generated.
     * It updates the bytecode and frame size.
     */
    void finish()
    {
      gen_.define_label(function_end_);
      gen_.define_relocatable(frame_size_, allocator_.frame_size());
    }

    RegisterAllocator& allocator()
    {
      return allocator_;
    }

    /**
     * Get a Relocatable that corresponds to the function's frame size, ie. the
     * total number of registers used.
     *
     * The actual value of this Relocatable isn't known until the function has
     * been generated entirely and `finish()` is called, since it depends on the
     * number of registers allocated.
     */
    Generator::Relocatable frame_size()
    {
      return frame_size_;
    }

    /**
     * Get a handle to a callee's register. This is useful to setup parameters
     * and access return values before and after a method call.
     *
     * The actual register index depends on the caller's frame size; therefore a
     * special CalleeRegister is used, which can be used with the `emit` method
     * anywhere a `Register` is expected.
     */
    CalleeRegister callee_register(const FunctionABI& callee_abi, uint8_t index)
    {
      return CalleeRegister(
        callee_abi.callspace(), frame_size_, Register(index));
    }

    Label create_label()
    {
      return gen_.create_label();
    }

    void define_label(Label label)
    {
      gen_.define_label(label);
    }

    Generator& writer()
    {
      return gen_;
    }

  private:
    Generator& gen_;
    RegisterAllocator allocator_;

    /**
     * Address of the end of the function. Used to compute the total function
     * size in the function header.
     */
    Label function_end_;

    /**
     * Total number of registers used by the function.
     * This is used in the function header and when accessing child-relative
     * registers.
     */
    Generator::Relocatable frame_size_;
  };

  /**
   * An extension to BaseFunctionGenerator for generating functions from some IR
   * representation.
   *
   * This class maintains mappings from IR variables and basic blocks to
   * registers and labels.
   */
  template<
    typename V = Variable,
    typename B = BasicBlock*,
    template<typename, typename, typename...> typename M = std::unordered_map>
  class FunctionGenerator : public BaseFunctionGenerator
  {
  public:
    FunctionGenerator(Generator& gen, std::string_view name, FunctionABI abi)
    : BaseFunctionGenerator(gen, name, abi)
    {}

    /**
     * Bind the function's parameter registers to the given SSA variables.
     *
     * Without this, calling the `variable` method on what should be function
     * parameters would allocate fresh registers. Binding the parameters allows
     * the variables to be tied to the correct registers.
     *
     * VS should be a collection of V or std::optional<V>. In the latter case,
     * any nullopt entry found does not create a mapping, but still consumes a
     * register.
     *
     * For example, binding the parameter list { nullopt, A, B } will map A to
     * r1 and B to r2.
     */
    template<typename VS>
    void bind_parameters(const VS& parameters)
    {
      assert(variables_.empty());
      uint32_t index = 0;
      for (std::optional<V> param : parameters)
      {
        if (param)
        {
          variables_.insert({*param, Register(index)});
        }
        index += 1;
      }
    }

    /**
     * Get the Register associated with the given SSA variable.
     *
     * Registers are allocated lazily when this function is first called for the
     * given basic block. Registers are currently never reused by other
     * variables.
     */
    Register variable(V var)
    {
      auto [it, inserted] = variables_.insert({var, bytecode::Register(0)});
      if (inserted)
        it->second = allocator().get();
      return it->second;
    }

    template<typename VS>
    std::vector<Register> variables(const VS& vars)
    {
      std::vector<bytecode::Register> result;
      for (auto v : vars)
      {
        result.push_back(variable(v));
      }
      return result;
    }

    /**
     * Get the Label associated with the given basic block's address.
     *
     * Labels are created lazily when this function is first called for the
     * given basic block.
     */
    Label label(B block)
    {
      auto [it, inserted] = labels.insert({block, Label()});
      if (inserted)
        it->second = create_label();
      return it->second;
    }

    using BaseFunctionGenerator::define_label;
    void define_label(B block)
    {
      BaseFunctionGenerator::define_label(label(block));
    }

  private:
    /**
     * A mapping from SSA variable to the corresponding allocated register.
     */
    M<V, bytecode::Register> variables_;

    /**
     * A mapping from basic block to the corresponding label.
     */
    M<B, Label> labels;
  };
}
