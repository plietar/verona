#pragma once

#include "Environment.h"
#include "Relation.h"

namespace mlir::verona
{
  template<typename... S>
  struct Join;

  /**
   * A selector combines a reference to a Relation with a tuple of values. Each
   * value can either be a placeholder or a concrete value.
   *
   * Selectors can be used to form queries on Relations, or to insert new
   * entries into the relation.
   *
   * For example, in `alias(_2, _1) += alias(_1, _2)`, both sides of the `+=`
   * operator are selectors.
   */
  template<typename R, typename... Ts>
  struct Selector
  {
    static_assert(
      is_valid_selector_v<std::tuple<Ts...>, typename R::tuple_type>);

    explicit Selector(R& relation, const std::tuple<Ts...>& pattern)
    : relation(relation), pattern(pattern)
    {}

    struct iterator;
    struct sentinel
    {};

    template<ExecutionMode mode = ExecutionMode::Stable>
    iterator begin() const
    {
      static_assert(is_valid_query_v<std::tuple<Ts...>, typename R::tuple_type>);

      auto lower = R::meta::template make_search_bound<lower_limit>(pattern);
      auto upper = R::meta::template make_search_bound<upper_limit>(pattern);

      const typename R::container_type* container;
      if constexpr (mode == ExecutionMode::Stable)
        container = &relation.get().stable_values;
      else
        container = &relation.get().recent_values;

      return iterator(
        pattern, container->lower_bound(lower), container->upper_bound(upper));
    }

    sentinel end() const
    {
      static_assert(is_valid_query_v<std::tuple<Ts...>, typename R::tuple_type>);
      return {};
    }

    template<typename E>
    using rebound_t = Selector<R, value_substitution_t<E, Ts>...>;

    template<typename... Es, typename = std::enable_if_t<is_valid_query_v<std::tuple<Ts...>, typename R::tuple_type, Environment<Es...>>>>
    rebound_t<Environment<Es...>> rebind(const Environment<Es...>& environment) const
    {
      auto refined_pattern =
        substitute<SubstituteLattice::No>(environment, pattern);
      return Selector<R, value_substitution_t<Environment<Es...>, Ts>...>(
        relation, refined_pattern);
    }

    struct iterator
    {
      using value_type =
        initial_environment_t<std::tuple<Ts...>, typename R::value_type>;

      value_type operator*() const
      {
        return make_environment(pattern, *current_it);
      }

      bool operator!=(sentinel)
      {
        return current_it != end_it;
      }

      iterator& operator++()
      {
        current_it++;
        return *this;
      }

    private:
      using underlying_iterator = typename R::container_type::const_iterator;

      explicit iterator(
        const std::tuple<Ts...>& pattern,
        underlying_iterator begin,
        underlying_iterator end)
      : pattern(pattern), current_it(begin), end_it(end)
      {}

      friend class Selector;

      std::tuple<Ts...> pattern;
      underlying_iterator current_it;
      underlying_iterator end_it;
    };

    template<typename... Es>
    void insert(const Environment<Es...>& environment) const
    {
      relation.get().insert(substitute(environment, pattern));
    }

    template<typename R2, typename... Ts2>
    auto join(Selector<R2, Ts2...> other)
    {
      return Join<Selector>(*this).join(other);
    }

  private:
    mutable std::reference_wrapper<R> relation;
    std::tuple<Ts...> pattern;
  };
}
