#pragma once

#include "Lattice.h"

#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace mlir::verona
{
  /**
   * Queries can be executed in two different modes: In Stable mode, only stable
   * results are considered. In Delta mode, only results that depend on at least
   * one recently computed value are considered.
   */
  enum class ExecutionMode
  {
    Stable,
    Delta,
  };

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
  struct is_valid_selector : std::false_type
  {};

  template<>
  struct is_valid_selector<std::tuple<>, std::tuple<>> : std::true_type
  {};

  /**
   * If the first element of the selector (T) is not a placeholder, it must be
   * the same as the first element of the relation.
   */
  template<typename T, typename... Ss, typename... Ts>
  struct is_valid_selector<
    std::tuple<T, Ss...>,
    std::tuple<T, Ts...>,
    std::enable_if_t<(std::is_placeholder_v<T> == 0)>>
  : is_valid_selector<std::tuple<Ss...>, std::tuple<Ts...>>
  {};

  /**
   * If the first element of the selector (P) is a placeholder, the first
   * element of the relation (T) can be anything.
   */
  template<typename P, typename T, typename... Ss, typename... Ts>
  struct is_valid_selector<
    std::tuple<P, Ss...>,
    std::tuple<T, Ts...>,
    std::enable_if_t<(std::is_placeholder_v<P>> 0)>>
  : is_valid_selector<std::tuple<Ss...>, std::tuple<Ts...>>
  {};

  template<typename Ss, typename Ts>
  static constexpr bool is_valid_selector_v = is_valid_selector<Ss, Ts>::value;

  template<typename Q, typename T, typename = void>
  struct is_valid_query : std::false_type
  {};

  template<>
  struct is_valid_query<std::tuple<>, std::tuple<>> : std::true_type
  {};

  template<typename Ph, typename T, typename... Qs, typename... Ts>
  struct is_valid_query<
    std::tuple<Ph, Qs...>,
    std::tuple<T, Ts...>,
    std::enable_if_t<std::is_placeholder_v<Ph> == 0 && !is_lattice_v<T>>>
  : is_valid_query<std::tuple<Qs...>, std::tuple<Ts...>>
  {};

  template<typename P, typename T, typename... Qs, typename... Ts>
  struct is_valid_query<
    std::tuple<P, Qs...>,
    std::tuple<T, Ts...>,
    std::enable_if_t<
      (std::is_placeholder_v<P>> 0) &&
      ((std::is_placeholder_v<Qs>> 0) && ...) &&
      ((std::is_placeholder_v<Qs> != std::is_placeholder_v<P>)&&...)>>
  : is_valid_query<std::tuple<Qs...>, std::tuple<Ts...>>
  {};

  template<typename Q, typename T, typename E = Environment<>>
  static constexpr bool is_valid_query_v = is_valid_query<substitution_t<E, Q>, T>::value;

  struct lower_limit
  {};
  struct upper_limit
  {};

  template<typename Compare>
  struct Index
  {
    using is_transparent = void;

    template<
      typename... Ts,
      typename... Us,
      typename = std::enable_if_t<sizeof...(Ts) == sizeof...(Us)>>
    bool operator()(
      const std::tuple<Ts...>& left, const std::tuple<Us...>& right) const
    {
      return compare_tuple<0>(left, right);
    }

    template<typename T, typename = std::enable_if_t<!is_tuple<T>::value>>
    bool operator()(const T& left, const T& right) const
    {
      return Compare()(left, right);
    }

    template<
      typename T,
      typename = std::enable_if_t<!std::is_same_v<T, lower_limit>>>
    bool operator()(lower_limit, const T&) const
    {
      return true;
    }

    template<
      typename T,
      typename = std::enable_if_t<!std::is_same_v<T, lower_limit>>>
    bool operator()(T&&, lower_limit) const
    {
      return false;
    }

    template<
      typename T,
      typename = std::enable_if_t<!std::is_same_v<T, upper_limit>>>
    bool operator()(upper_limit, T&&) const
    {
      return false;
    }

    template<
      typename T,
      typename = std::enable_if_t<!std::is_same_v<T, upper_limit>>>
    bool operator()(T&&, upper_limit) const
    {
      return true;
    }

    template<size_t I, typename... Ts, typename... Us>
    bool compare_tuple(
      const std::tuple<Ts...>& left, const std::tuple<Us...>& right) const
    {
      static_assert(sizeof...(Ts) == sizeof...(Us));
      if constexpr (I < sizeof...(Ts))
      {
        using T = std::tuple_element_t<I, std::tuple<Ts...>>;
        using U = std::tuple_element_t<I, std::tuple<Us...>>;

        const T& l = std::get<I>(left);
        const U& r = std::get<I>(right);
        if ((*this)(l, r))
          return true;
        else if ((*this)(r, l))
          return false;
        else
          return compare_tuple<I + 1>(left, right);
      }
      else
      {
        return false;
      }
    }
  };

  template<typename T, size_t I = 0, typename = void>
  struct relation_indices;

  template<size_t I>
  struct relation_indices<std::tuple<>, I>
  {
    using key_indices = std::index_sequence<>;
    using mapped_indices = std::index_sequence<>;
  };

  template<typename T, typename... Ts, size_t I>
  struct relation_indices<
    std::tuple<T, Ts...>,
    I,
    std::enable_if_t<is_lattice_v<T>>>
  {
    using base = relation_indices<std::tuple<Ts...>, I + 1>;
    using key_indices = typename base::key_indices;
    using mapped_indices =
      index_sequence_cons_t<I, typename base::mapped_indices>;

    static_assert(
      base::key_indices::size() == 0,
      "Lattice types may not be interleaved with non-lattice ones.");
  };

  template<typename T, typename... Ts, size_t I>
  struct relation_indices<
    std::tuple<T, Ts...>,
    I,
    std::enable_if_t<!is_lattice_v<T>>>
  {
    using base = relation_indices<std::tuple<Ts...>, I + 1>;
    using key_indices = index_sequence_cons_t<I, typename base::key_indices>;
    using mapped_indices = typename base::mapped_indices;
  };

  template<
    typename T,
    typename = typename relation_indices<T>::key_indices,
    typename = typename relation_indices<T>::mapped_indices>
  struct relation_types;

  template<typename Tuple, size_t... Keys, size_t... Mapped>
  struct relation_types<
    Tuple,
    std::index_sequence<Keys...>,
    std::index_sequence<Mapped...>>
  {
    static constexpr size_t Arity = std::tuple_size_v<Tuple>;
    using key_type = std::tuple<std::tuple_element_t<Keys, Tuple>...>;
    using mapped_type = std::tuple<std::tuple_element_t<Mapped, Tuple>...>;
    using value_type = std::pair<key_type, mapped_type>;

    static_assert(Arity == sizeof...(Keys) + sizeof...(Mapped));

    static value_type assemble_value(Tuple tuple)
    {
      key_type key = std::make_tuple(std::get<Keys>(tuple)...);
      mapped_type mapped = std::make_tuple(std::get<Mapped>(tuple)...);
      return std::make_pair(key, mapped);
    }

    template<
      typename B,
      typename Pattern,
      typename = std::enable_if_t<(std::tuple_size_v<Pattern> == Arity)>,
      typename = std::enable_if_t<
        ((std::is_placeholder_v<std::tuple_element_t<Mapped, Pattern>>> 0) &&
         ...)>>
    static auto make_search_bound(const Pattern& pattern)
    {
      return std::make_tuple(make_search_bound_impl<B, Keys>(pattern)...);
    }

  private:
    template<typename B, size_t I, typename Pattern>
    static auto make_search_bound_impl(const Pattern& pattern)
    {
      using T = std::tuple_element_t<I, Tuple>;
      using U = std::tuple_element_t<I, Pattern>;
      if constexpr (std::is_placeholder_v<U>> 0)
        return B();
      else
        return T(std::get<I>(pattern));
    }
  };

  template<typename R, typename... Ts>
  struct Selector;

  template<typename Tuple, typename Compare = std::less<>>
  struct Relation
  {
    using meta = relation_types<Tuple>;
    using tuple_type = Tuple;

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

    template<typename... T>
    void emplace(T&&... values)
    {
      pending_values.push_back(
        meta::assemble_value(std::make_tuple(values...)));
    }

    void insert(Tuple key)
    {
      pending_values.push_back(meta::assemble_value(key));
    }

    template<
      typename... Ts,
      typename =
        std::enable_if_t<is_valid_selector_v<std::tuple<Ts...>, Tuple>>>
    Selector<Relation, Ts...> operator()(Ts... pattern)
    {
      return Selector<Relation, Ts...>(*this, std::make_tuple(pattern...));
    }

    Relation() = default;
    Relation(const Relation& other) = delete;

  private:
    using key_type = typename meta::key_type;
    using mapped_type = typename meta::mapped_type;
    using value_type = typename meta::value_type;
    using container_type = std::map<key_type, mapped_type, Index<Compare>>;

    container_type stable_values;
    container_type recent_values;
    std::vector<value_type> pending_values;

    template<typename R, typename... Ts>
    friend struct Selector;
  };
}
