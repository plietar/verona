
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
   * An Environment maps indices to concrete values. The mapped indices need not
   * be contiguous, and different indices may map to values of different types.
   *
   * Which indices are present in the environment and what type each index maps
   * to is reflected in the instantiation of the Environment template. For
   * example, an environment with signature { 0: int, 1: char, 3: float } would
   * be represented by the C++ type Environment<int, char, void, float>. Notice
   * the use of `void` to indicate the absence of value at index 2.
   *
   * The typical way of constructing an environment is by binding a query with
   * its result, using the `make_environment` function. For each placeholder
   * value found in the query (ie. values from std::placeholders, such as _1,
   * _2, ...), the corresponding value in the result is inserted in the map.
   *
   * Once an environment is constructed, it can be used to substitute the same
   * placeholders values in other tuples, using the `substitute` function.
   *
   * Note that placeholders are 1-indexed, whereas Environment's methods are
   * 0-indexed. For example, substituting placeholder _2 using an environment
   * { 0 => A, 1 => B, 2 => C } will return value `B`. This is generally not an
   * issue, as the only two functions that relate placeholders to environments,
   * `make_environment` and `substitute`, take care are mapping from one to
   * another.
   */
  template<typename... Ts>
  struct Environment
  {
    // We forbid any trailing void in Ts, as that would allow different
    // representations for the same environments, eg. Environment<int> and
    // Environment<int, void>.
    //
    // We could have canonical_enviroment_t<Ts...> definition to trim trailing
    // voids and give the right Environment<> instantiation, but there doesn't
    // seem to be a need for it so far.
    static_assert(
      !has_trailing_void_v<Ts...>,
      "Environment must not be instantiated with trailing `void` types");

    // In practice, we can't construct a tuple with void values. Therefore we
    // replace any void within Ts... by environment_hole. It's not the most
    // efficient, since each environment_hole may still occupy some space.
    template<typename T>
    using replace_void =
      std::conditional_t<std::is_void_v<T>, environment_hole, T>;

    Environment(std::tuple<replace_void<Ts>...> values) : values_(values) {}

    /**
     * Determine the type of the element at index I, or void is no value is
     * present for that index.
     *
     * This may be used for any index, including beyond the length of Ts. In the
     * latter case, element_type will always be void.
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
     * Get the value at a given index. The environment is statically required to
     * contain a value for that particular index.
     */
    template<size_t I>
    std::enable_if_t<contains<I>, element_type<I>> get() const
    {
      return std::get<I>(values_);
    }

    /**
     * An environment is contiguous if the indices it maps to are contiguous and
     * start at zero.
     */
    static constexpr bool is_contiguous = (!std::is_void_v<Ts> && ...);

    /**
     * For contiguous environments, this method allows direct access to the
     * underlying tuple representation.
     */
    std::conditional_t<is_contiguous, const std::tuple<Ts...>&, void>
    values() const
    {
      if constexpr (is_contiguous)
      {
        return values_;
      }
    }

    using tuple_type =
      std::conditional_t<is_contiguous, std::tuple<Ts...>, void>;

  private:
    std::tuple<replace_void<Ts>...> values_;
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

  template<size_t P, typename Q, size_t N = 0>
  struct placeholder_indices;

  template<size_t P, size_t N>
  struct placeholder_indices<P, std::tuple<>, N>
  {
    static_assert(P > 0, "Placeholders indices start at 1");
    using type = std::index_sequence<>;
  };

  template<size_t P, typename Q, typename... Qs, size_t N>
  struct placeholder_indices<P, std::tuple<Q, Qs...>, N>
  {
    static_assert(P > 0, "Placeholders indices start at 1");
    using base =
      typename placeholder_indices<P, std::tuple<Qs...>, N + 1>::type;
    using type = std::conditional_t<
      (std::is_placeholder_v<Q> == P),
      index_sequence_cons_t<N, base>,
      base>;
  };

  /**
   * Find all occurences of placeholder P in the tuple Q. The computed type is
   * an instantiation of std::index_sequence, encoding the index of each
   * occurrence.
   */
  template<size_t P, typename Q>
  using placeholder_indices_t = typename placeholder_indices<P, Q>::type;

  template<typename R>
  environment_hole lookup_result(const R& results, std::index_sequence<>)
  {
    return environment_hole();
  }

  template<typename... Rs, size_t I>
  auto lookup_result(const std::tuple<Rs...>& results, std::index_sequence<I>)
  {
    return std::get<I>(results);
  }

  template<typename... Rs1, typename... Rs2, size_t I>
  auto lookup_result(
    const std::pair<const std::tuple<Rs1...>, std::tuple<Rs2...>>& results,
    std::index_sequence<I>)
  {
    if constexpr (I < sizeof...(Rs1))
      return std::get<I>(results.first);
    else
      return std::get<I - sizeof...(Rs1)>(results.second);
  }

  template<typename... Rs1, typename... Rs2, size_t I>
  auto lookup_result(
    const std::pair<std::tuple<Rs1...>, std::tuple<Rs2...>>& results,
    std::index_sequence<I>)
  {
    if constexpr (I < sizeof...(Rs1))
      return std::get<I>(results.first);
    else
      return std::get<I - sizeof...(Rs1)>(results.second);
  }

  /**
   * Create an environment by binding a query to its results.
   *
   * For each occurrence of a placeholder in the query, the corresponding
   * concrete value from the results is added to the environment.
   *
   * For instance, binding the query `(A, _1)` together with the result `(A, B)`
   * produces the environment { 0 => B }.
   *
   * `results` should either be a tuple of the same length as query, or
   * `pair<const tuple<Ts...>, T>`. The latter is used for Relations that are
   * internally represented as a std::map.
   */
  template<typename... Qs, typename R>
  auto make_environment(std::tuple<Qs...> query, const R& results)
  {
    constexpr size_t N = std::max({0UL, std::is_placeholder_v<Qs>...});
    return Environment(generate_tuple<N>([&](auto I) {
      using Indices = placeholder_indices_t<I + 1, std::tuple<Qs...>>;
      return lookup_result(results, Indices());
    }));
  }

  enum class SubstituteLattice
  {
    Yes,
    No
  };

  /**
   * Apply the environment to a value, replacing placeholders with the
   * corresponding value from the environment. If no such value exists, the
   * placeholder is returned as is.
   */
  template<
    SubstituteLattice substitute_lattice = SubstituteLattice::Yes,
    typename... Ts,
    typename U>
  auto substitute_value(const Environment<Ts...>& environment, const U& value)
  {
    constexpr size_t P = std::is_placeholder_v<U>;
    if constexpr (P > 0 && Environment<Ts...>::template contains<P - 1>)
    {
      using Element = typename Environment<Ts...>::template element_type<P - 1>;
      if constexpr (
        substitute_lattice == SubstituteLattice::Yes || !is_lattice_v<Element>)
      {
        return environment.template get<P - 1>();
      }
      else
        return value;
    }
    else
      return value;
  }

  /**
   * Apply an environment to a tuple of values, replacing all placeholders by
   * their corresponding values from the environment. Placeholders for which no
   * value exists in the environment are left unmodified.
   */
  template<
    SubstituteLattice substitute_lattice = SubstituteLattice::Yes,
    typename... Ts,
    typename... Us>
  auto substitute(
    const Environment<Ts...>& environment, const std::tuple<Us...>& values)
  {
    constexpr size_t N = sizeof...(Us);
    return generate_tuple<N>([&](auto I) {
      return substitute_value<substitute_lattice>(
        environment, std::get<I>(values));
    });
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

  /**
   * Determine whether two Environment instantiations can be unified.
   * Environments can be unified if the only indices they overlap on are lattice
   * types.
   */
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

  template<
    typename E,
    typename T,
    SubstituteLattice substitute_lattice = SubstituteLattice::Yes>
  using substitution_t = decltype(substitute<substitute_lattice>(
    std::declval<const E&>(), std::declval<const T&>()));

  template<
    typename E,
    typename T,
    SubstituteLattice substitute_lattice = SubstituteLattice::Yes>
  using value_substitution_t = decltype(substitute_value<substitute_lattice>(
    std::declval<const E&>(), std::declval<const T&>()));

  void test_environment()
  {
    static_assert(Environment<int, void, float>::contains<0>);
    static_assert(!Environment<int, void, float>::contains<1>);
    static_assert(Environment<int, void, float>::contains<2>);
    static_assert(!Environment<int, void, float>::contains<3>);
  }

  template<size_t P, typename Tuple, typename Expected>
  void test_placeholder_indices()
  {
    static_assert(std::is_same_v<placeholder_indices_t<P, Tuple>, Expected>);
  }

  void test_placeholder_indices()
  {
    using namespace std;
    using PH1 = std::remove_cv_t<decltype(std::placeholders::_1)>;
    using PH2 = std::remove_cv_t<decltype(std::placeholders::_2)>;

    test_placeholder_indices<1, tuple<>, index_sequence<>>();
    test_placeholder_indices<1, tuple<>, index_sequence<>>();
    test_placeholder_indices<2, tuple<>, index_sequence<>>();
    test_placeholder_indices<1, tuple<int>, index_sequence<>>();
    test_placeholder_indices<1, tuple<PH1>, index_sequence<0>>();
    test_placeholder_indices<2, tuple<PH1>, index_sequence<>>();
    test_placeholder_indices<1, tuple<PH1, PH1>, index_sequence<0, 1>>();
    test_placeholder_indices<2, tuple<PH1, PH1>, index_sequence<>>();
    test_placeholder_indices<1, tuple<PH2, PH1>, index_sequence<1>>();
    test_placeholder_indices<2, tuple<PH2, PH1>, index_sequence<0>>();
    test_placeholder_indices<1, tuple<PH2, int, PH1>, index_sequence<2>>();
    test_placeholder_indices<2, tuple<PH2, int, PH1>, index_sequence<0>>();
    test_placeholder_indices<2, tuple<PH2, int, PH2>, index_sequence<0, 2>>();
  }

  template<typename Q, typename R, typename Expected>
  void test_initial_environment()
  {
    static_assert(std::is_same_v<initial_environment_t<Q, R>, Expected>);
  }

  void test_initial_environment()
  {
    using namespace std;
    using PH1 = std::remove_cv_t<decltype(std::placeholders::_1)>;
    using PH2 = std::remove_cv_t<decltype(std::placeholders::_2)>;

    test_initial_environment<tuple<>, tuple<>, Environment<>>();
    test_initial_environment<tuple<int>, tuple<int>, Environment<>>();
    test_initial_environment<tuple<PH1>, tuple<int>, Environment<int>>();
    test_initial_environment<tuple<PH2>, tuple<int>, Environment<void, int>>();

    test_initial_environment<
      tuple<int, PH2>,
      tuple<int, int>,
      Environment<void, int>>();

    test_initial_environment<
      tuple<PH2, PH1>,
      tuple<char, int>,
      Environment<int, char>>();
  }

  template<typename E1, typename E2, bool Expected>
  void test_can_unify()
  {
    static_assert(can_unify_v<E1, E2> == Expected);
    static_assert(can_unify_v<E2, E1> == Expected);
  }

  void test_can_unify()
  {
    /*
    test_can_unify<Environment<>, Environment<int>, true>();
    test_can_unify<Environment<char>, Environment<void, int>, true>();

    test_can_unify<Environment<unit>, Environment<unit>, false>();
    test_can_unify<
      Environment<lattice<unit>>,
      Environment<lattice<int>>,
      false>();
    test_can_unify<
      Environment<lattice<unit>>,
      Environment<lattice<unit>>,
      true>();
    */

    /*
    test_can_unify<Environment<int>, Environment<int>, false>();
    test_can_unify<Environment<int, int>, Environment<void, int>, true>();
    test_can_unify<Environment<char>, Environment<int>, false>();
    test_can_unify<Environment<int, char>, Environment<int, int>, false>();
    test_can_unify<Environment<int, char, int>, Environment<int, int>, false>();
    test_can_unify<Environment<int, char, int>, Environment<char>, false>();
    */
  }

  template<typename E1, typename E2, typename Expected>
  void test_unified_environment()
  {
    // static_assert(std::is_same_v<unified_environment_t<E1, E2>, Expected>);
    // static_assert(std::is_same_v<unified_environment_t<E2, E1>, Expected>);
  }

  void test_unified_environment()
  {
    test_unified_environment<
      Environment<>,
      Environment<int>,
      Environment<int>>();

    test_unified_environment<
      Environment<char>,
      Environment<void, int>,
      Environment<char, int>>();

    test_unified_environment<
      Environment<int, int>,
      Environment<void, int>,
      Environment<int, int>>();

    test_unified_environment<
      Environment<char, int>,
      Environment<void, int>,
      Environment<char, int>>();
  }

  template<typename E, typename T, typename Expected>
  void test_substitution()
  {
    static_assert(std::is_same_v<substitution_t<E, T>, Expected>);
  }

  void test_substitution()
  {
    using namespace std;
    using PH1 = std::remove_cv_t<decltype(std::placeholders::_1)>;
    using PH2 = std::remove_cv_t<decltype(std::placeholders::_2)>;

    test_substitution<Environment<>, tuple<int, char>, tuple<int, char>>();
    test_substitution<Environment<int>, tuple<int, char>, tuple<int, char>>();
    test_substitution<Environment<int>, tuple<PH1, char>, tuple<int, char>>();
    test_substitution<Environment<int>, tuple<PH1, PH2>, tuple<int, PH2>>();

    /*
    test_substitution<
      Environment<int, char>,
      tuple<PH1, PH2>,
      tuple<int, char>>();

    test_substitution<
      Environment<void, char>,
      tuple<PH1, PH2>,
      tuple<PH1, char>>();
    */
  }
}
