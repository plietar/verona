// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/reachability/reachability.h"

#include "compiler/analysis.h"
#include "compiler/format.h"
#include "compiler/ir/ir.h"
#include "compiler/recursive_visitor.h"
#include "compiler/resolution.h"
#include "compiler/typecheck/solver.h"
#include "compiler/typecheck/typecheck.h"

#include <fmt/ostream.h>
#include <map>
#include <queue>

namespace verona::compiler
{
  typedef std::variant<CodegenItem<Entity>, CodegenItem<Method>>
    ReachabilityItem;

  struct ReachabilityVisitor
  : private RecursiveTypeVisitor<const Instantiation&>
  {
    ReachabilityVisitor(
      Context& context, const Program& program, const AnalysisResults& analysis)
    : context_(context),
      program_(program),
      analysis_(analysis),
      solver_out_(context_.dump("reachability-solver"))
    {}

    void process(CodegenItem<Method> main)
    {
      // The class must be processed before the method
      push(main.parent());
      push(main);

      while (!queue_.empty())
      {
        ReachabilityItem item = queue_.front();
        queue_.pop();

        auto [_, inserted] = visited_.insert(item);
        if (inserted)
        {
          std::visit([&](const auto& inner) { visit_item(inner); }, item);
        }
      }
    }

    /**
     * Push a new item to be processed for reachability.
     */
    void push(ReachabilityItem item)
    {
      if (visited_.find(item) == visited_.end())
      {
        queue_.push(item);
      }
    }

    void visit_item(const CodegenItem<Entity>& item)
    {
      // If there's already a reachable item equivalent to this one we don't
      // need to do anything.
      if (find_equivalence(item))
        return;

      EntityReachability& info = add_entity(item);
      visit_generic_bounds(*item.definition->generics, item.instantiation);
      visit_fields_and_final(item);

      consider_entity_subtyping(item, info);
    }

    void visit_fields_and_final(const CodegenItem<Entity>& item)
    {
      for (const auto& member : item.definition->members)
      {
        if (const Field* fld = member->get_as<Field>())
        {
          result_.selectors.insert(Selector::field(fld->name));
          visit_type(fld->type, item.instantiation);
        }
        // Finalisers are reachable, if the type is reachable.
        if (const Method* mtd = member->get_as<Method>())
        {
          if (mtd->is_finaliser())
          {
            // Finalisers never have type parameters, so we can reuse item's
            // instantiation
            CodegenItem<Method> method_item(mtd, item.instantiation);
            visit_item(method_item);
          }
        }
      }
    }

    /**
     * Find an already reachable item that is equivalent to `new_item`.
     *
     * If such an equivalent entity is found, the equivalence is recorded in the
     * result and true is returned.
     */
    bool find_equivalence(const CodegenItem<Entity>& new_item)
    {
      for (auto& [other, _] : result_.entities)
      {
        if (check_equivalent(new_item, other))
        {
          result_.equivalent_entities.insert({new_item, other});
          return true;
        }
      }
      return false;
    }

    /**
     * For the given entity, find all other reachable entities that are sub-
     * or supertypes of that one, and update the reachability info with any of
     * these relationship.
     */
    void consider_entity_subtyping(
      const CodegenItem<Entity>& item, EntityReachability& item_info)
    {
      for (auto& [other, other_info] : result_.entities)
      {
        if (item == other)
          continue;

        consider_subtyping_pair(other, item, item_info);
        consider_subtyping_pair(item, other, other_info);
      }
    }

    /**
     * Given a pair of entities, check if `sub` is a subtype of `super`.
     *
     * If so, add this relationship to the result and push to the queue any
     * method that is reachable in `super`, but applied to `sub`.
     */
    void consider_subtyping_pair(
      const CodegenItem<Entity>& sub,
      const CodegenItem<Entity>& super,
      EntityReachability& super_info)
    {
      if (check_subtype(sub, super))
      {
        super_info.subtypes.insert(sub);

        // Any already reachable method in super is now reachable in sub,
        // and needs to be added to the queue.
        for (const auto& method : super_info.methods)
        {
          push(sub.method(method.selector()));
        }
      }
    }

    /**
     * Check whether one entity is a subtype of another.
     */
    bool check_subtype(
      const CodegenItem<Entity>& sub, const CodegenItem<Entity>& super)
    {
      Constraint constraint(
        context_.mk_entity_type(sub.definition, sub.instantiation.types()),
        context_.mk_entity_type(super.definition, super.instantiation.types()),
        0,
        context_);

      Solver solver(context_, *solver_out_);
      Solver::SolutionSet solutions =
        solver.solve_one(constraint, SolverMode::Verify);
      solver.print_stats(solutions);

      return !solutions.empty();
    }

    /**
     * Check whether two entities are equivalent to each other.
     */
    bool check_equivalent(
      const CodegenItem<Entity>& left, const CodegenItem<Entity>& right)
    {
      TypePtr left_ty =
        context_.mk_entity_type(left.definition, left.instantiation.types());
      TypePtr right_ty =
        context_.mk_entity_type(right.definition, right.instantiation.types());

      Solver solver(context_, *solver_out_);
      Solver::SolutionSet solutions = solver.solve_all(
        {Constraint(left_ty, right_ty, 0, context_),
         Constraint(right_ty, left_ty, 0, context_)},
        SolverMode::Verify);
      solver.print_stats(solutions);

      return !solutions.empty();
    }

    void visit_item(const CodegenItem<Method>& item)
    {
      CodegenItem<Entity> parent = item.parent();
      Selector selector = item.selector();

      if (auto it = result_.equivalent_entities.find(parent);
          it != result_.equivalent_entities.end())
      {
        // Make the method reachable on the equivalent entity instead.
        push(it->second.method(selector));
        return;
      }

      result_.selectors.insert(selector);

      EntityReachability& parent_info = result_.entities.at(parent);
      parent_info.add_method(item);

      visit_signature(item.definition->signature->types, item.instantiation);
      visit_generic_bounds(
        *item.definition->signature->generics, item.instantiation);

      if (item.definition->body)
        visit_body(analysis_.functions.at(item.definition), item.instantiation);

      // Also add that method for any already known subtype
      for (const auto& subtype : parent_info.subtypes)
      {
        push(subtype.method(selector));
      }
    }

    void visit_signature(
      const TypeSignature& signature, const Instantiation& instantiation)
    {
      visit_type(signature.receiver, instantiation);
      visit_types(signature.arguments, instantiation);
      visit_type(signature.return_type, instantiation);
    }

    void visit_generic_bounds(
      const Generics& generics, const Instantiation& instantiation)
    {
      for (const auto& param : generics.types)
      {
        visit_type(param->bound, instantiation);
      }
    }

    void
    visit_body(const FnAnalysis& analysis, const Instantiation& instantiation)
    {
      for (const auto& ir : (*analysis.ir).function_irs)
      {
        IRTraversal traversal(*ir);
        while (BasicBlock* bb = traversal.next())
        {
          const TypeAssignment& assignment = analysis.typecheck->types.at(bb);

          for (const auto& stmt : bb->statements)
          {
            std::visit(
              [&](const auto& s) {
                visit_stmt(s, instantiation, *analysis.typecheck, assignment);
              },
              stmt);
          }

          std::visit(
            [&](const auto& t) { visit_term(t, instantiation); },
            *bb->terminator);
        }
      }
    }

    void visit_entity_type(
      const EntityTypePtr& ty, const Instantiation& instantiation) final
    {
      TypeList arguments = instantiation.apply(context_, ty->arguments);
      push(CodegenItem(ty->definition, Instantiation(arguments)));

      visit_types(arguments, Instantiation::empty());
    }

    void visit_static_type(
      const StaticTypePtr& ty, const Instantiation& instantiation) final
    {
      TypeList arguments = instantiation.apply(context_, ty->arguments);
      push(CodegenItem(ty->definition, Instantiation(arguments)));

      visit_types(arguments, Instantiation::empty());
    }

    /**
     * Visitor that walks a Type looking for methods with the given name.
     *
     * For each entity or static type of the disjunction / conjunction with a
     * method of that name, the method is added to the parent
     * ReachabilityVisitor's queue, with the right instantiation.
     */
    struct CallReachability : public TypeVisitor<>
    {
      CallReachability(
        ReachabilityVisitor* parent,
        const std::string& method_name,
        const TypeList& call_arguments)
      : parent(parent), method_name(method_name), call_arguments(call_arguments)
      {}

      void visit_entity(const Entity* entity, const TypeList& entity_arguments)
      {
        if (const Method* method = lookup_member<Method>(entity, method_name))
        {
          // We need to build an instantiation that combines the type
          // arguments applied to the entity with the type arguments applied
          // to the method.
          TypeList arguments = entity_arguments;
          std::copy(
            call_arguments.begin(),
            call_arguments.end(),
            std::back_inserter(arguments));
          parent->push(CodegenItem(entity, Instantiation(entity_arguments)));
          parent->push(CodegenItem(method, Instantiation(arguments)));
        }
      }

      void visit_static_type(const StaticTypePtr& ty) final
      {
        visit_entity(ty->definition, ty->arguments);
      }

      void visit_entity_type(const EntityTypePtr& ty) final
      {
        visit_entity(ty->definition, ty->arguments);
      }

      void visit_union(const UnionTypePtr& ty) final
      {
        visit_types(ty->elements);
      }

      void visit_intersection(const IntersectionTypePtr& ty) final
      {
        visit_types(ty->elements);
      }

      void visit_variable_renaming_type(const VariableRenamingTypePtr& ty) final
      {}

      void visit_capability(const CapabilityTypePtr& ty) final {}

      ReachabilityVisitor* parent;
      const std::string& method_name;
      const TypeList& call_arguments;
    };

    void visit_stmt(
      const CallStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {
      TypePtr receiver =
        instantiation.apply(context_, assignment.at(stmt.receiver));

      TypeList arguments = instantiation.apply(
        context_, typecheck.type_arguments.at(stmt.type_arguments));

      visit_types(arguments, Instantiation::empty());

      CallReachability v(this, stmt.method, arguments);
      v.visit_type(receiver);
    }

    void visit_stmt(
      const StaticTypeStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {
      TypeList arguments = instantiation.apply(
        context_, typecheck.type_arguments.at(stmt.type_arguments));
      push(CodegenItem(stmt.definition, Instantiation(arguments)));
    }

    void visit_stmt(
      const NewStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {
      TypeList arguments = instantiation.apply(
        context_, typecheck.type_arguments.at(stmt.type_arguments));
      push(CodegenItem(stmt.definition, Instantiation(arguments)));
    }

    void visit_stmt(
      const MatchBindStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {}

    void visit_stmt(
      const ReadFieldStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {
      result_.selectors.insert(Selector::field(stmt.name));
    }

    void visit_stmt(
      const WriteFieldStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {
      result_.selectors.insert(Selector::field(stmt.name));
    }

    void visit_stmt(
      const CopyStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {}

    void visit_stmt(
      const WhenStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {
      // No reachability required, as closure has been lifted out.
    }

    void visit_stmt(
      const IntegerLiteralStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {
      push(CodegenItem(program_.find_entity("U64"), Instantiation::empty()));
    }

    void visit_stmt(
      const StringLiteralStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {}

    void visit_stmt(
      const UnitStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {}

    void visit_stmt(
      const ViewStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {}

    void visit_stmt(
      const EndScopeStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {}

    void visit_stmt(
      const OverwriteStmt& stmt,
      const Instantiation& instantiation,
      const TypecheckResults& typecheck,
      const TypeAssignment& assignment)
    {}

    void
    visit_term(const MatchTerminator& term, const Instantiation& instantiation)
    {
      for (const auto& arm : term.arms)
      {
        visit_type(arm.type, instantiation);
      }
    }

    void
    visit_term(const BranchTerminator& term, const Instantiation& instantiation)
    {}

    void
    visit_term(const ReturnTerminator& term, const Instantiation& instantiation)
    {}

    void
    visit_term(const IfTerminator& term, const Instantiation& instantiation)
    {}

    EntityReachability& add_entity(const CodegenItem<Entity>& entity)
    {
      EntityReachability reachability;
      auto [it, inserted] = result_.entities.insert({entity, reachability});
      if (!inserted)
        throw std::logic_error("Entity added multiple times");

      return it->second;
    }

    Context& context_;
    const Program& program_;
    const AnalysisResults& analysis_;
    Reachability result_;

    std::queue<ReachabilityItem> queue_;
    std::set<ReachabilityItem> visited_;
    std::unique_ptr<std::ostream> solver_out_;
  };

  const CodegenItem<Entity>&
  Reachability::normalize_equivalence(const CodegenItem<Entity>& entity) const
  {
    auto it = equivalent_entities.find(entity);
    if (it != equivalent_entities.end())
      return it->second;
    else
      return entity;
  }

  EntityReachability&
  Reachability::find_entity(const CodegenItem<Entity>& entity)
  {
    return entities.at(normalize_equivalence(entity));
  }

  const EntityReachability&
  Reachability::find_entity(const CodegenItem<Entity>& entity) const
  {
    return entities.at(normalize_equivalence(entity));
  }

  bool Reachability::is_reachable(const CodegenItem<Entity>& entity) const
  {
    auto it = entities.find(normalize_equivalence(entity));
    return it != entities.end();
  }

  void dump_reachability(Context& context, const Reachability& reachability)
  {
    auto output = context.dump("reachability");
    for (const auto& [entity, info] : reachability.entities)
    {
      fmt::print(*output, "{} {}\n", entity.definition->kind->value(), entity);
      for (const auto& method : info.methods)
      {
        fmt::print(*output, "  method {}\n", method);
      }
      for (const auto& subtype : info.subtypes)
      {
        fmt::print(*output, "  subtype {}\n", subtype);
      }
    }
    for (const auto& [entity, equivalent] : reachability.equivalent_entities)
    {
      fmt::print(
        *output,
        "{} {} => {} {}\n",
        entity.definition->kind->value(),
        entity,
        equivalent.definition->kind->value(),
        equivalent);
    }
    for (const auto& selector : reachability.selectors)
    {
      fmt::print(*output, "selector {}\n", selector);
    }
  }

  Reachability compute_reachability(
    Context& context,
    const Program& program,
    CodegenItem<Method> main,
    const AnalysisResults& analysis)
  {
    ReachabilityVisitor v(context, program, analysis);
    v.process(main);

    dump_reachability(context, v.result_);

    return v.result_;
  }

  bool is_valid_main_signature(Context& context, const FnSignature& signature)
  {
    return signature.generics->types.empty() && signature.receiver == nullptr &&
      signature.types.arguments.empty() &&
      signature.types.return_type == context.mk_unit();
  }

  std::optional<CodegenItem<Method>>
  find_program_entry(Context& context, const Program& program)
  {
    const Entity* main_class = program.find_entity("Main");
    if (!main_class)
    {
      report(
        context, std::nullopt, DiagnosticKind::Error, Diagnostic::NoMainClass);
      return std::nullopt;
    }

    if (main_class->kind->value() != Entity::Class)
    {
      report(
        context,
        main_class->name.source_range,
        DiagnosticKind::Error,
        Diagnostic::MainNotAClass);
      return std::nullopt;
    }

    if (!main_class->generics->types.empty())
    {
      report(
        context,
        main_class->name.source_range,
        DiagnosticKind::Error,
        Diagnostic::MainClassIsGeneric);
      return std::nullopt;
    }

    const Method* main_method = lookup_member<Method>(main_class, "main");
    if (!main_method)
    {
      report(
        context,
        main_class->name.source_range,
        DiagnosticKind::Error,
        Diagnostic::NoMainMethod);
      return std::nullopt;
    }

    if (!is_valid_main_signature(context, *main_method->signature))
    {
      report(
        context,
        main_method->name.source_range,
        DiagnosticKind::Error,
        Diagnostic::InvalidMainSignature);
      return std::nullopt;
    }

    return CodegenItem(main_method, Instantiation::empty());
  }

  std::ostream& operator<<(std::ostream& s, const Selector& selector)
  {
    return s << selector.name << format::optional_list(selector.arguments);
  }
}
