#include "compiler/codegen/item.h"

#include "compiler/resolution.h"

namespace verona::compiler
{
  CodegenItem<Method>
  CodegenItem<Entity>::method(const Selector& selector) const
  {
    const Method* method =
      lookup_member<Method>(this->definition, selector.name);
    if (method == nullptr)
      throw std::logic_error("Method missing.");

    TypeList arguments = this->instantiation.types();
    arguments.insert(
      arguments.end(), selector.arguments.begin(), selector.arguments.end());
    return CodegenItem<Method>(method, Instantiation(arguments));
  }

  CodegenItem<Entity> CodegenItem<Method>::parent() const
  {
    TypeList arguments;
    for (const auto& param : this->definition->parent->generics->types)
    {
      arguments.push_back(this->instantiation.types().at(param->index));
    }
    return CodegenItem<Entity>(
      this->definition->parent, Instantiation(arguments));
  }

  Selector CodegenItem<Method>::selector() const
  {
    TypeList arguments;
    for (const auto& param : this->definition->signature->generics->types)
    {
      arguments.push_back(this->instantiation.types().at(param->index));
    }
    return Selector::method(this->definition->name, arguments);
  }

  CodegenItem<Entity> CodegenItem<Field>::parent() const
  {
    // Fields don't have type parameters of their own, so its instantiation will
    // be the same as its parent's
    return CodegenItem<Entity>(this->definition->parent, this->instantiation);
  }

  Selector CodegenItem<Field>::selector() const
  {
    return Selector::field(this->definition->name);
  }

  std::ostream& operator<<(std::ostream& s, const CodegenItem<Method>& item)
  {
    return s << item.definition->instantiated_path(item.instantiation);
  }

  std::ostream& operator<<(std::ostream& s, const CodegenItem<Entity>& item)
  {
    return s << item.definition->instantiated_path(item.instantiation);
  }
}
