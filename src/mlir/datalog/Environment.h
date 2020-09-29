
#pragma once

#include "Lattice.h"
#include "helpers.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>

namespace mlir::verona
{
  struct environment_hole
  {};

  /**
   * An Environment maps placeholder values (eg. _1, _2, ...) to concrete
   * values.
   *
   * It is constructed by combining a query with its corresponding concrete
   * results. For example, if the query `foo(A, _1)` gives result `(A, B)`, the
   * resulting environment would be `{ _1 => B }`. Since the first value is
   * already concrete in the query, its corresponding result is not included in
   * the environment.
   *
   * Internally, the environment is represented as a tuple of values. The nth
   * element of the tuple represents the value of placeholder n. If a
   * placeholder has no value in the environment (eg. the environment assigns
   * values to _1 and _3 but not _2), the corresponding element in the tuple
   * would be void. The type of the environment statically reflects which
   * placeholders are bound or not.
   *
   * For example, the environment { _1 => B, _3 => C } is represented by
   * (B, _, C). It would have type `Environment<T, void, U>`, where T and U
   * are the types of concrete values B and C.
   */
  template<typename... Ts>
  struct Environment
  {
    // In practice, we can't construct a tuple with void values. Therefore we
    // replace any void within Ts... by environment_hole.
    template<typename T>
    using replace_void =
      std::conditional_t<std::is_void_v<T>, environment_hole, T>;

    Environment(std::tuple<replace_void<Ts>...> values) : values_(values) {}

    /**
     * Determine the type of the element at index I, or void is no value is
     * present for that index.
     *
     * This may be used for any index, beyond the length of Ts. Anything index
     * greater or equal to sizeof...(Ts) will return void.
     */
    template<size_t I>
    using element_type =
      typename std::conditional_t < I<sizeof...(Ts),
                                      std::tuple_element<I, std::tuple<Ts...>>,
                                      std::common_type<void>>::type;

    /**
     * True if the environment holds a value for the given index.
     */
    template<size_t I>
    static constexpr bool contains = !std::is_void_v<element_type<I>>;

    /**
     * Get the value at a given index.
     */
    template<size_t I>
    std::enable_if_t<contains<I>, element_type<I>> get() const
    {
      return std::get<I>(values_);
    }

    /**
     * Get the upper bound for valid indices.
     */
    static constexpr size_t size_bound = sizeof...(Ts);

    /**
     * An environment is complete if all indices up to size_bound are defined.
     */
    static constexpr bool is_complete = (!std::is_void_v<Ts> && ...);

    using tuple_type = std::conditional_t<is_complete, std::tuple<Ts...>, void>;
    tuple_type values() const
    {
      if constexpr (is_complete)
      {
        return values_;
      }
    }

    /**
     * Apply the environment to a tuple of values, replacing all placeholders by
     * their corresponding values from the environment.
     */
    template<typename... Us>
    auto substitute(const std::tuple<Us...>& t) const
    {
      return substitute_impl(std::index_sequence_for<Us...>(), t);
    }

    /**
     * Apply the environment to a value, replacing placeholders with the
     * corresponding value from the environment. If no such value exists, the
     * placeholder is returned as is.
     */
    template<typename U>
    auto substitute_value(const U& x) const
    {
      constexpr size_t P = std::is_placeholder_v<U>;
      if constexpr (P > 0 && contains<P - 1>)
        return get<P - 1>();
      else
        return x;
    }

  private:
    std::tuple<replace_void<Ts>...> values_;

    template<size_t... I, typename... Us>
    auto
    substitute_impl(std::index_sequence<I...>, const std::tuple<Us...>& t) const
    {
      return std::make_tuple(substitute_value(std::get<I>(t))...);
    }
  };

  template<typename... Ts>
  Environment(const std::tuple<Ts...>& values)
    ->Environment<
      std::conditional_t<std::is_same_v<Ts, environment_hole>, void, Ts>...>;

  /**
   * Create an empty environment.
   */
  Environment<> empty_environment()
  {
    return Environment<>({});
  }

  template<size_t P, typename Q, typename = void>
  struct find_placeholder
  {
    static constexpr std::optional<size_t> value = std::nullopt;
  };

  template<size_t P, typename Q, typename = void>
  static constexpr std::optional<size_t> find_placeholder_v =
    find_placeholder<P, Q>::value;

  template<size_t P, typename Q, typename... Qs>
  struct find_placeholder<
    P,
    std::tuple<Q, Qs...>,
    std::enable_if_t<std::is_placeholder_v<Q> == P>>
  {
    static constexpr std::optional<size_t> value = 0;
  };

  template<size_t P, typename Q, typename... Qs>
  struct find_placeholder<
    P,
    std::tuple<Q, Qs...>,
    std::enable_if_t<
      std::is_placeholder_v<Q> != P &&
      find_placeholder_v<P, std::tuple<Qs...>>.has_value()>>
  {
    static constexpr std::optional<size_t> value =
      1 + *find_placeholder_v<P, std::tuple<Qs...>>;
  };

  template<
    size_t I,
    typename... Rs,
    typename = std::enable_if_t<I<sizeof...(Rs)>> auto uniform_indexing(
      const std::tuple<Rs...>& results)
  {
    return std::get<I>(results);
  }

  template<
    size_t I,
    typename... Rs,
    typename Rl,
    typename = std::enable_if_t<I<sizeof...(Rs) + 1>> auto uniform_indexing(
      const std::pair<const std::tuple<Rs...>, Rl>& results)
  {
    if constexpr (I < sizeof...(Rs))
      return std::get<I>(results.first);
    else
      return results.second;
  }

  /**
   * Create an environment by binding a query with its results.
   *
   * For each occurrence of a placeholder in the query, the corresponding
   * concrete value from the results is added to the environment.
   *
   * TODO: this doesn't properly handle multiple occurrences of the same value.
   */
  template<typename... Qs, typename R>
  auto make_environment(std::tuple<Qs...> query, const R& results)
  {
    constexpr size_t N = std::max({0UL, std::is_placeholder_v<Qs>...});
    return Environment(generate_tuple<N>([&](auto I) {
      constexpr std::optional<size_t> offset =
        find_placeholder_v<I + 1, std::tuple<Qs...>>;

      if constexpr (offset.has_value())
        return uniform_indexing<*offset>(results);
      else
        return environment_hole();
    }));
  }

  template<typename E1, typename E2>
  struct can_unify;
  template<typename... Ts, typename... Us>
  struct can_unify<Environment<Ts...>, Environment<Us...>>
  {
    static constexpr size_t N = std::max(sizeof...(Ts), sizeof...(Us));
    static constexpr bool value = forall<N>([](auto I) {
      using T = typename Environment<Ts...>::template element_type<I>;
      using U = typename Environment<Us...>::template element_type<I>;
      return std::is_void_v<T> || std::is_void_v<U> || std::is_same_v<T, U>;
    });
  };
  template<typename E1, typename E2>
  static constexpr bool can_unify_v = can_unify<E1, E2>::value;

  template<
    typename... Ts,
    typename... Us,
    typename =
      std::enable_if_t<can_unify_v<Environment<Ts...>, Environment<Us...>>>>
  auto unify(const Environment<Ts...>& left, const Environment<Us...>& right)
  {
    constexpr size_t N = std::max(sizeof...(Ts), sizeof...(Us));

    return Environment(generate_tuple<N>([&](auto I) {
      using T = typename Environment<Ts...>::template element_type<I>;
      using U = typename Environment<Us...>::template element_type<I>;

      if constexpr (!std::is_void_v<T> && !std::is_void_v<U>)
      {
        static_assert(std::is_same_v<T, U>);
        const T& left_value = left.template get<I>();
        const U& right_value = right.template get<I>();

        if constexpr (is_lattice_v<T>)
        {
          return T::lub(left_value, right_value);
        }
        else
        {
          assert(left_value == right_value);
          return left_value;
        }
      }
      else if constexpr (!std::is_void_v<T>)
        return left.template get<I>();
      else if constexpr (!std::is_void_v<U>)
        return right.template get<I>();
      else
        return environment_hole();
    }));
  }

  template<typename Q, typename R>
  using initial_environment_t = decltype(
    make_environment(std::declval<const Q&>(), std::declval<const R&>()));

  template<typename E1, typename E2>
  using unified_environment_t =
    decltype(unify(std::declval<const E1&>(), std::declval<const E2&>()));

  void test_environment()
  {
    static_assert(Environment<int, void, float>::contains<0>);
    static_assert(!Environment<int, void, float>::contains<1>);
    static_assert(Environment<int, void, float>::contains<2>);
    static_assert(!Environment<int, void, float>::contains<3>);
  }

  void test_find_placeholders()
  {
    using namespace std::placeholders;
    static_assert(find_placeholder_v<1, std::tuple<>> == std::nullopt);
    static_assert(find_placeholder_v<2, std::tuple<>> == std::nullopt);
    static_assert(find_placeholder_v<1, std::tuple<int>> == std::nullopt);
    static_assert(find_placeholder_v<1, std::tuple<decltype(_1)>> == 0);
    static_assert(
      find_placeholder_v<1, std::tuple<decltype(_2), decltype(_1)>> == 1);
    static_assert(
      find_placeholder_v<2, std::tuple<decltype(_2), decltype(_1)>> == 0);
    static_assert(
      find_placeholder_v<3, std::tuple<decltype(_2), decltype(_1)>> ==
      std::nullopt);
  }

  void test_initial_environment()
  {
    using namespace std::placeholders;

    static_assert(std::is_same_v<
                  initial_environment_t<std::tuple<>, std::tuple<>>,
                  Environment<>>);

    static_assert(
      std::is_same_v<
        initial_environment_t<std::tuple<decltype(_1)>, std::tuple<int>>,
        Environment<int>>);
  }

  void test_unify()
  {
    enum class A
    {
    };
    enum class B
    {
    };

    static_assert(can_unify_v<Environment<>, Environment<A>>);
    static_assert(can_unify_v<Environment<B>, Environment<void, A>>);
    static_assert(can_unify_v<Environment<A, A>, Environment<void, A>>);
    static_assert(can_unify_v<Environment<B, A>, Environment<void, A>>);

    static_assert(!can_unify_v<Environment<B>, Environment<A>>);
    static_assert(!can_unify_v<Environment<A, B, A>, Environment<A, A, A>>);
    static_assert(!can_unify_v<Environment<A, B, A>, Environment<A, A>>);
    static_assert(!can_unify_v<Environment<A, B, A>, Environment<B>>);

    static_assert(std::is_same_v<
                  unified_environment_t<Environment<>, Environment<A>>,
                  Environment<A>>);
    static_assert(std::is_same_v<
                  unified_environment_t<Environment<B>, Environment<void, A>>,
                  Environment<B, A>>);
    static_assert(
      std::is_same_v<
        unified_environment_t<Environment<A, A>, Environment<void, A>>,
        Environment<A, A>>);
    static_assert(
      std::is_same_v<
        unified_environment_t<Environment<B, A>, Environment<void, A>>,
        Environment<B, A>>);
  }
}
