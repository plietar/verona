// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/instantiation.h"
#include "compiler/reachability/item.h"
#include "compiler/reachability/selector.h"
#include "compiler/resolution.h"

#include <unordered_set>

/**
 * The reachability phase determines the set of items that need to be
 * codegenerated.
 */
namespace verona::compiler
{
  struct AnalysisResults;

  struct EntityReachability
  {
    std::vector<CodegenItem<Method>> methods;
    std::optional<CodegenItem<Method>> finaliser;

    // Set of reified entities which are subtypes of this one. It will only be
    // non-empty for interfaces.
    std::set<CodegenItem<Entity>> subtypes;

    void add_method(const CodegenItem<Method>& method)
    {
      if (method.definition->is_finaliser())
        finaliser = method;

      methods.push_back(method);
    }
  };

  struct Reachability
  {
    std::map<CodegenItem<Entity>, EntityReachability> entities;
    std::set<Selector> selectors;

    /**
     * There can be multiple equivalent entities that are reachable from the
     * program. In this case we pick a canonical one (the first one we come
     * across) and make the others point to it in this map.
     *
     * Only the canonical entity will be included in the program.
     */
    std::map<CodegenItem<Entity>, CodegenItem<Entity>> equivalent_entities;

    /**
     * Find the canonical item that is equivalent to `entity`.
     */
    const CodegenItem<Entity>&
    normalize_equivalence(const CodegenItem<Entity>& entity) const;

    /**
     * Find the information related to this entity or an equivalent one.
     */
    EntityReachability& find_entity(const CodegenItem<Entity>& entity);
    const EntityReachability&
    find_entity(const CodegenItem<Entity>& entity) const;

    bool is_reachable(const CodegenItem<Entity>& entity) const;
  };

  Reachability compute_reachability(
    Context& context,
    const Program& program,
    CodegenItem<Method> main,
    const AnalysisResults& analysis);

  /**
   * Search for the program entrypoint and check it has the right signature.
   *
   * Returns nullopt and raises errors in the context if the entrypoint isn't
   * found or is invalid.
   */
  std::optional<CodegenItem<Method>>
  find_program_entry(Context& context, const Program& program);
}
