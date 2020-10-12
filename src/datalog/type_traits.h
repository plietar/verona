#pragma once

#include "datalog/environment.h"

namespace datalog
{
  /**
   * Type trait used to verify the type of a Selector against the type of a
   * Relation. The first argument, S, is the std::tuple of the elements of the
   * selector, where as the second, C, is the std::tuple formed of the concrete
   * types used in the relation.
   *
   * A selector is valid for a given relation if they have the same arity and if
   * each element of the selector is either a placeholder or has the same type
   * as the corresponding element of the relation.
   *
   * Note that selectors appear both on the left and the right of Horn clauses.
   * The rules dictated by `is_valid_selector` are therefore fairly limited.
   * Another trait, `is_valid_query`, is used for selectors that appear on the
   * right hand side of clauses.
   *
   * The trait is defined inductively on the size of the tuples.
   */
  template<typename S, typename C, typename = void>
  struct is_valid_selector : std::false_type
  {};

  template<>
  struct is_valid_selector<std::tuple<>, std::tuple<>> : std::true_type
  {};

  /**
   * If the first element of the selector (T) is not a placeholder, it must be
   * the same as the first element of the relation.
   */
  template<typename T, typename... Ss, typename... Cs>
  struct is_valid_selector<
    std::tuple<T, Ss...>,
    std::tuple<T, Cs...>,
    std::enable_if_t<(std::is_placeholder_v<T> == 0)>>
  : is_valid_selector<std::tuple<Ss...>, std::tuple<Cs...>>
  {};

  /**
   * If the first element of the selector (P) is a placeholder, the first
   * element of the relation (C) can be anything.
   */
  template<typename P, typename... Ss, typename C, typename... Cs>
  struct is_valid_selector<
    std::tuple<P, Ss...>,
    std::tuple<C, Cs...>,
    std::enable_if_t<(std::is_placeholder_v<P> != 0)>>
  : is_valid_selector<std::tuple<Ss...>, std::tuple<Cs...>>
  {};

  template<typename S, typename C>
  static constexpr bool is_valid_selector_v = is_valid_selector<S, C>::value;

  /**
   * Type trait used to verify the type of a query against the type of a
   * Relation. The rules are slightly stricter than those of general selectors:
   * placeholders cannot be repeated, a placeholder may not be followed by a
   * concrete type, and lattice values may not be matched against concrete
   * types.
   */
  template<typename Q, typename C, typename = void>
  struct is_valid_query : std::false_type
  {};

  template<>
  struct is_valid_query<std::tuple<>, std::tuple<>> : std::true_type
  {};

  template<typename T, typename... Qs, typename... Cs>
  struct is_valid_query<
    std::tuple<T, Qs...>,
    std::tuple<T, Cs...>,
    std::enable_if_t<std::is_placeholder_v<T> == 0 && !is_lattice_v<T>>>
  : is_valid_query<std::tuple<Qs...>, std::tuple<Cs...>>
  {};

  template<typename P, typename... Qs, typename C, typename... Ts>
  struct is_valid_query<
    std::tuple<P, Qs...>,
    std::tuple<C, Ts...>,
    std::enable_if_t<
      (std::is_placeholder_v<P> != 0) &&
      ((std::is_placeholder_v<Qs> != 0) && ...) &&
      ((std::is_placeholder_v<Qs> != std::is_placeholder_v<P>)&&...)>>
  : is_valid_query<std::tuple<Qs...>, std::tuple<Ts...>>
  {};

  template<typename Q, typename T>
  static constexpr bool is_valid_query_v = is_valid_query<Q, T>::value;
}
