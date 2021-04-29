#pragma once

#include "compiler/ast.h"
#include "compiler/reachability/selector.h"

namespace verona::compiler
{
  /**
   * A reference to an AST node with a specific instantiation.
   */
  template<typename T>
  struct BaseCodegenItem
  {
    const T* definition;
    Instantiation instantiation;

    explicit BaseCodegenItem(const T* definition, Instantiation instantiation)
    : definition(definition), instantiation(instantiation)
    {}

    std::string instantiated_path() const
    {
      return definition->instantiated_path(instantiation);
    }

    bool operator<(const BaseCodegenItem<T>& other) const
    {
      return std::tie(definition, instantiation) <
        std::tie(other.definition, other.instantiation);
    }

    bool operator==(const BaseCodegenItem<T>& other) const
    {
      return std::tie(definition, instantiation) ==
        std::tie(other.definition, other.instantiation);
    }
  };

  template<typename T>
  struct CodegenItem;

  template<>
  struct CodegenItem<Entity> : BaseCodegenItem<Entity>
  {
    using BaseCodegenItem<Entity>::BaseCodegenItem;

    CodegenItem<Method> method(const Selector& selector) const;
  };

  template<>
  struct CodegenItem<Method> : BaseCodegenItem<Method>
  {
    using BaseCodegenItem<Method>::BaseCodegenItem;

    CodegenItem<Entity> parent() const;
    Selector selector() const;
  };

  template<>
  struct CodegenItem<Field> : BaseCodegenItem<Field>
  {
    using BaseCodegenItem<Field>::BaseCodegenItem;

    CodegenItem<Entity> parent() const;
    Selector selector() const;
  };

  template<typename T>
  CodegenItem(const T* definition, Instantiation instantiation)->CodegenItem<T>;

  std::ostream& operator<<(std::ostream& s, const CodegenItem<Method>& item);
  std::ostream& operator<<(std::ostream& s, const CodegenItem<Entity>& item);
}
