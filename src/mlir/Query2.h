#pragma once

// #include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace mlir::verona
{
  template<size_t N>
  struct placeholder
  {
    static_assert(N > 0);
  };
}

template<size_t N>
struct std::is_placeholder<mlir::verona::placeholder<N>>
{
  static constexpr size_t value = N;
};

namespace mlir::verona
{
  struct lower_limit
  {};
  struct upper_limit
  {};

  struct Index
  {
    using is_transparent = void;

    template<typename... Ts, typename... Us>
    bool operator()(
      const std::tuple<Ts...>& left, const std::tuple<Us...>& right) const
    {
      return compare<0>(left, right);
    }

    template<typename T>
    bool compare_element(const T& left, const T& right) const
    {
      return std::less<>()(left, right);
    }

    bool compare_element(lower_limit, lower_limit) const
    {
      return false;
    }

    template<typename T>
    bool compare_element(lower_limit, const T&) const
    {
      return true;
    }

    template<typename T>
    bool compare_element(T&&, lower_limit) const
    {
      return false;
    }

    template<typename T>
    bool compare_element(upper_limit, T&&) const
    {
      return false;
    }

    bool compare_element(upper_limit, upper_limit) const
    {
      return false;
    }

    template<typename T>
    bool compare_element(T&&, upper_limit) const
    {
      return true;
    }

    template<size_t I, typename... Ts, typename... Us>
    bool
    compare(const std::tuple<Ts...>& left, const std::tuple<Us...> right) const
    {
      static_assert(sizeof...(Ts) == sizeof...(Us));
      if constexpr (I < sizeof...(Ts))
      {
        using T = std::tuple_element_t<I, std::tuple<Ts...>>;
        using U = std::tuple_element_t<I, std::tuple<Us...>>;

        const T& l = std::get<I>(left);
        const U& r = std::get<I>(right);
        if (compare_element(l, r))
          return true;
        else if (compare_element(r, l))
          return false;
        else
          return compare<I + 1>(left, right);
      }
      else
      {
        return false;
      }
    }
  };

  // template<size_t P, typename... Qs>
  // constexpr std::optional<size_t> find_placeholder();

  template<size_t P>
  constexpr std::optional<size_t> find_placeholder()
  {
    return std::nullopt;
  }

  template<size_t P, typename Q, typename... Qs>
  constexpr std::optional<size_t> find_placeholder()
  {
    static_assert(P > 0);

    if constexpr (std::is_placeholder_v<Q> == P)
      return 0;
    else if constexpr (constexpr auto idx = find_placeholder<P, Qs...>())
      return *idx + 1;
    else
      return std::nullopt;
  }

  static_assert(find_placeholder<1>() == std::nullopt);
  static_assert(find_placeholder<1, int>() == std::nullopt);
  static_assert(find_placeholder<1, placeholder<1>>() == 0);
  static_assert(find_placeholder<1, placeholder<2>, placeholder<1>>() == 1);
  static_assert(find_placeholder<2, placeholder<2>, placeholder<1>>() == 0);
  static_assert(
    find_placeholder<3, placeholder<2>, placeholder<1>>() == std::nullopt);

  /**
   * Type trait used to verify the type of a Selector with the type of a
   * Relation.
   *
   * A selector is valid for a given relation if they have the same arity and if
   * each element of the selector is either a placeholder or has the same type
   * as the corresponding element of the relation.
   *
   * The trait is defined inductively on the size of the tuples.
   */
  template<typename S, typename C, typename = void>
  struct typecheck_selectors : std::false_type
  {};

  template<>
  struct typecheck_selectors<std::tuple<>, std::tuple<>> : std::true_type
  {};

  /**
   * If the first element of the selector (Ph) is a placeholder, the first
   * element of the relation (T) can be anything.
   */
  template<typename Ph, typename T, typename... S, typename... C>
  struct typecheck_selectors<
    std::tuple<Ph, S...>,
    std::tuple<T, C...>,
    std::enable_if_t<(std::is_placeholder_v<Ph>> 0)>>
  : typecheck_selectors<std::tuple<S...>, std::tuple<C...>>
  {};

  /**
   * If the first element of the selector (T) is not a placeholder, it must be
   * the same as the first element of the relation.
   */
  template<typename T, typename... S, typename... C>
  struct typecheck_selectors<
    std::tuple<T, S...>,
    std::tuple<T, C...>,
    std::enable_if_t<(std::is_placeholder_v<T> == 0)>>
  : typecheck_selectors<std::tuple<S...>, std::tuple<C...>>
  {};

  /**
   * An Environment maps placeholder values (eg. _1, _2, ...) to concrete
   * values.
   *
   * It is constructed (and extended) by combining a query with the
   * corresponding concrete results of the query. For example, if the query
   * `foo(A, _1)` gives result `(A, B)`, the resulting environment would be
   * `{ _1 => B }`. Since the first value is already concrete in the query, its
   * corresponding result is not used to build the environment.
   *
   * Internally, the environment is represented as a tuple of values. The nth
   * element of the tuple represents the value of placeholder n. If a
   * placeholder has no value in the environment (eg. the environment assigns
   * values to _1 and _3 but not _2), the corresponding element in the tuple
   * would be the placeholder itself. The type of the environment statically
   * reflects which placeholders are bound or not.
   *
   * For example, the environment { _1 => B, _3 => C } is represented by
   * (B, _2, C). It would have type `Environment<T, ph<2>, U>`, where T and U
   * are the types of concrete values B and C, and ph<2> is the type of _2.
   */
  template<typename... Ts>
  struct Environment
  {
    /**
     * Create an empty environment. This constructor is only available if
     * Ts is empty.
     */
    template<
      bool enable = sizeof...(Ts) == 0,
      typename = std::enable_if_t<enable>>
    explicit Environment()
    {}

    /**
     * Returns true if placeholder P has an associated value in the current
     * environment.
     */
    template<size_t P>
    static constexpr bool has_value()
    {
      if constexpr (P <= sizeof...(Ts))
        return std::is_placeholder_v<std::tuple_element_t<P-1, std::tuple<Ts...>>> == 0;
      else
        return false;
    }


    /**
     * Extend an environment using a query tuple and its corresponding result.
     *
     * For any placeholder in the query, the corresponding result is added to
     * the environment, if not yet present. If a concrete value already exists
     * in the environment for this placeholder, the new and existing value must
     * be equal.
     *
     * For example extending environment { _1 => B, _3 => C } using query
     * (_2, _1, D) and result (A, B, D) will produce environment
     * { _1 => B, _2 => A, _3 => C }.
     */
    template<typename... Qs, typename... Rs>
    auto extend(
      const std::tuple<Qs...>& query, const std::tuple<Rs...>& results) const
    {
      static_assert(
        sizeof...(Qs) == sizeof...(Rs),
        "A query and its results should have the same arity");
      static_assert(
        typecheck_selectors<std::tuple<Qs...>, std::tuple<Rs...>>::value);

      // The result environment needs to be big enough to hold all values in the
      // existing environment (N), as well as any placeholder found in Qs.
      constexpr size_t N = sizeof...(Ts);
      constexpr size_t M = std::max({N, std::is_placeholder_v<Qs>...});

      return extend_impl(std::make_index_sequence<M>(), query, results);
    }

    /**
     * Use the environment to substitute any placeholders in the given tuple.
     * Placeholders than are undefined in the environment are left as is.
     *
     * For example, the environment { _1 => B, _3 => C } applied to tuple
     * (A, _1, _2) gives result (A, B, _2).
     */
    template<typename... Us>
    auto substitute(const std::tuple<Us...>& t) const
    {
      return substitute_impl(std::index_sequence_for<Us...>(), t);
    }

    /**
     * Lookup a value in the environment, if the argument is a placeholder. If
     * the placeholder is unbound in the environment, the same placeholder is
     * returned. Similarily, if the argument is a concrete value, it is returned
     * unmodified.
     */
    template<typename U>
    auto substitute_value(const U& x) const
    {
      constexpr size_t P = std::is_placeholder_v<U>;
      if constexpr (P > 0 && P <= sizeof...(Ts))
        return std::get<P - 1>(values);
      else
        return x;
    }

    /**
     * Use the environment to invoke the given callable object. The values of
     * placeholders are used in numerical order as the arguments of the
     * callable. The environment must be complete.
     *
     * For example, applying a function to an environment { _1 => A, _2 => B }
     * will invoke the function with arguments (A, B).
     *
     * The result of the invocation is returned.
     */
    template<typename Fn>
    std::invoke_result_t<Fn&&, const Ts&...> apply(Fn&& fn) const
    {
      static_assert(is_complete);
      return std::apply(std::forward<Fn>(fn), values);
    }

    /**
     * An environment is complete if there are no gaps in the placeholders it
     * maps. For instance { _1 => B, _2 => C } is complete, but
     * { _1 => B, _3 => C } is not.
     */
    static constexpr bool is_complete =
      ((std::is_placeholder_v<Ts> == 0) && ...);

  private:
    template<size_t... I, typename... Us>
    auto
    substitute_impl(std::index_sequence<I...>, const std::tuple<Us...>& t) const
    {
      return std::make_tuple(substitute_value(std::get<I>(t))...);
    }

    template<size_t... I, typename... Qs, typename... Rs>
    auto extend_impl(
      std::index_sequence<I...>,
      const std::tuple<Qs...>& query,
      const std::tuple<Rs...>& results) const
    {
      return make_environment(
        std::make_tuple(extend_value<I + 1>(query, results)...));
    }

    /**
     * Get the new value of placeholder P in the extended environment.
     */
    template<size_t P, typename... Qs, typename... Rs>
    auto extend_value(
      const std::tuple<Qs...>& query, const std::tuple<Rs...>& results) const
    {
      static_assert(P > 0);
      constexpr std::optional<size_t> idx = find_placeholder<P, Qs...>();

      if constexpr (idx.has_value() && has_value<P>())
      {
        const auto& new_value = std::get<*idx>(results);
        const auto& old_value = std::get<P - 1>(values);
        assert(new_value == old_value);
        return new_value;
      }
      else if constexpr (idx.has_value())
      {
        return std::get<*idx>(results);
      }
      else if constexpr (has_value<P>())
      {
        return std::get<P - 1>(values);
      }
      else
      {
        return placeholder<P>();
      }
    }

    explicit Environment(std::tuple<Ts...> values) : values(values) {}

    template<typename... Us>
    friend class Environment;

    /**
     * Create another Environment with (possibly) different template arguments.
     * This method is necessary to deduce the new template arguments.
     *
     * Using just `Environment(...)` would not work as this refers to the same
     * instantiation of the Environment class, rather than deducing new
     * parameters.
     */
    template<typename... Us>
    Environment<Us...> make_environment(std::tuple<Us...> values) const
    {
      return Environment<Us...>(values);
    }

    std::tuple<Ts...> values;
  };

  template<typename... S>
  struct Join;
  template<typename Fn, typename... S>
  struct Rule;

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
      sizeof...(Ts) == std::tuple_size_v<typename R::tuple_type>,
      "The selector must have the same arity as the relation it refers to");
    static_assert(
      typecheck_selectors<std::tuple<Ts...>, typename R::tuple_type>::value);

    R& relation;
    std::tuple<Ts...> values;

    Selector(R& relation, std::tuple<Ts...> values)
    : relation(relation), values(values)
    {}

    template<typename R2, typename... Ts2>
    Join<Selector<R, Ts...>, Selector<R2, Ts2...>>
    join(Selector<R2, Ts2...> other) const
    {
      return {std::make_tuple(*this, other)};
    }

    template<typename R2, typename... Ts2>
    Join<Selector<R, Ts...>, Selector<R2, Ts2...>>
    operator&(Selector<R2, Ts2...> other) const
    {
      return {std::make_tuple(*this, other)};
    }

    template<typename J>
    void operator+=(const J& join)
    {
      std::apply(
        [&](auto... y) {
          auto f = std::bind(
            [](const auto&... x) { return typename R::key_type(x...); }, y...);
          relation += join.with(f);
        },
        values);
    }
  };

  template<typename... S>
  struct Join
  {
    static_assert(sizeof...(S) < 64);
    std::tuple<S...> selectors;

    template<typename Q>
    Join<S..., Q> join(Q other) const
    {
      return {std::tuple_cat(selectors, std::make_tuple(other))};
    }

    template<typename Fn>
    void execute(Fn&& fn) const
    {
      execute_impl<0, 0>(fn, Environment());
    }

    template<uint64_t Recent = 1, typename Fn>
    void execute_delta(Fn&& fn) const
    {
      static_assert(Recent > 0);
      if constexpr (Recent < 1 << sizeof...(S))
      {
        execute_impl<0, Recent>(fn, Environment());
        execute_delta<Recent + 1>(fn);
      }
    }

    template<typename Fn>
    Rule<Fn, S...> with(Fn fn) const
    {
      return Rule<Fn, S...>{fn, *this};
    }

  private:
    template<size_t I, uint64_t Recent, typename Fn, typename Env>
    void execute_impl(const Fn& fn, const Env& env) const
    {
      if constexpr (I == sizeof...(S))
      {
        env.apply(fn);
      }
      else
      {
        const auto& selector = std::get<I>(selectors);
        auto pattern = env.substitute(selector.values);

        if constexpr ((Recent & (1 << I)) == 0)
        {
          auto [begin, end] = selector.relation.search_stable(pattern);
          for (auto it = begin; it != end; it++)
          {
            execute_impl<I + 1, Recent>(fn, env.extend(pattern, *it));
          }
        }
        else
        {
          auto [begin, end] = selector.relation.search_recent(pattern);
          for (auto it = begin; it != end; it++)
          {
            execute_impl<I + 1, Recent>(fn, env.extend(pattern, *it));
          }
        }
      }
    }
  };

  template<typename Fn, typename... S>
  struct Rule
  {
    Fn fn;
    Join<S...> join;

    template<typename Cb>
    void compute_delta(Cb&& cb) const
    {
      join.execute_delta([&](auto... values) { cb(fn(values...)); });
    }
  };

  template<typename Key, typename Compare>
  struct IndexedLattice
  {
    using tuple_type = Key;
    using key_type = Key;
    bool iterate()
    {
      for (const auto& value : recent_values)
      {
        auto [it, inserted] = stable_values.insert(value);
        // if (!inserted)
        //   it->second = value.second;
      }
      recent_values.clear();

      for (const auto& value : pending_values)
      {
        auto [it, inserted] = recent_values.insert(value);
        /*
        TODO: optimise if stable value is already "as good" as pending one.
        if (!inserted)
        {
          it->second = lattice::lub(it->second, value.second);
        }
        else if (auto stable_it = stable_values.find(value.first);
                 stable_it != stable_values.end())
        {
          it->second = lattice::lub(it->second, stable_it->second);
        }
        */
      }
      pending_values.clear();

      return !recent_values.empty();
    }

    template<typename B, typename T>
    static auto make_bound(const T& value)
    {
      if constexpr (std::is_placeholder_v<T>> 0)
        return B();
      else
        return value;
    }

    template<typename B, typename... Ts, size_t... Is>
    static auto
    make_bound(const std::tuple<Ts...>& values, std::index_sequence<Is...>)
    {
      return std::make_tuple(make_bound<B>(std::get<Is>(values))...);
    }

    template<typename... Ts>
    static auto search_in(
      const std::set<Key, Compare>& values, const std::tuple<Ts...>& pattern)
    {
      auto lower =
        make_bound<lower_limit>(pattern, std::index_sequence_for<Ts...>());
      auto upper =
        make_bound<upper_limit>(pattern, std::index_sequence_for<Ts...>());
      return std::make_pair(
        values.lower_bound(lower), values.upper_bound(upper));
    }

    template<typename... Ts>
    auto search_stable(const std::tuple<Ts...>& pattern) const
    {
      return search_in(stable_values, pattern);
    }

    template<typename... Ts>
    auto search_recent(const std::tuple<Ts...>& pattern) const
    {
      return search_in(recent_values, pattern);
    }

    template<typename... Ts>
    Selector<IndexedLattice, Ts...> operator()(Ts... keys)
    {
      return Selector(*this, std::make_tuple(keys...));
    }

    void add(Key key)
    {
      pending_values.push_back(key);
    }

    template<typename Fn, typename... S>
    void operator+=(const Rule<Fn, S...>& rule)
    {
      rule.compute_delta([&](Key entry) { add(entry); });
    }

    std::set<Key, Compare> stable_values;
    std::set<Key, Compare> recent_values;

    std::vector<Key> pending_values;
  };
}
