#pragma once

#include "datalog/environment.h"
#include "datalog/relation.h"

namespace datalog
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

    template<ExecutionMode mode = ExecutionMode::Stable, typename R_ = R>
    std::enable_if_t<
      std::is_same_v<R_, R> &&
        is_valid_query_v<std::tuple<Ts...>, typename R_::tuple_type>,
      iterator>
    begin() const
    {
      static_assert(
        is_valid_query_v<std::tuple<Ts...>, typename R::tuple_type>);

      return iterator(
        pattern,
        relation.get().template lower_bound<mode>(pattern),
        relation.get().template upper_bound<mode>(pattern));
    }

    sentinel end() const
    {
      return {};
    }

    template<
      typename E,
      typename = std::enable_if_t<is_valid_selector_v<
        substitution_t<E, std::tuple<Ts...>, SubstituteLattice::No>,
        typename R::tuple_type>>>
    auto rebind(const E& environment) const
    {
      auto refined_pattern =
        substitute<SubstituteLattice::No>(environment, pattern);
      return make_selector(relation.get(), refined_pattern);
    }

    struct iterator
    {
      using underlying_iterator = typename R::iterator;
      using value_type = make_environment_t<
        std::tuple<Ts...>,
        typename underlying_iterator::value_type>;

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

    template<typename R2, typename... Ts2>
    auto join(Selector<R2, Ts2...> other)
    {
      return Join<Selector>(*this).join(other);
    }

    template<typename Fn>
    auto with(Fn&& fn)
    {
      return Join<Selector>(*this).with(std::forward<Fn>(fn));
    }

    template<typename... S>
    auto operator+=(const Join<S...>& join)
    {
      relation.get() +=
        join.with([&](auto env) { return substitute(env, pattern); });
    }

  private:
    mutable std::reference_wrapper<R> relation;
    std::tuple<Ts...> pattern;

    template<typename R2, typename... T2>
    static Selector<R2, T2...>
    make_selector(R2& relation, const std::tuple<T2...>& pattern)
    {
      return Selector<R2, T2...>(relation, pattern);
    }
  };

  template<typename E, typename S>
  using rebind_selector_t =
    decltype(std::declval<const S&>().rebind(std::declval<const E&>()));
}
