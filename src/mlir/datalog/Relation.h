#pragma once

#include "Lattice.h"
#include "Selector.h"

#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace mlir::verona
{
  struct lower_limit
  {};
  struct upper_limit
  {};

  template<typename Compare>
  struct Index
  {
    using is_transparent = void;

    template<typename>
    struct is_tuple : std::false_type
    {};
    template<typename... T>
    struct is_tuple<std::tuple<T...>> : std::true_type
    {};

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

  template<
    typename Tuple,
    typename Compare,
    typename = std::make_index_sequence<std::tuple_size_v<Tuple> - 1>,
    typename = void>
  struct RelationMeta
  {
    using key_type = Tuple;
    using value_type = Tuple;
    using container_type = std::set<value_type, Index<Compare>>;

    static constexpr size_t arity = std::tuple_size_v<Tuple>;

    template<
      typename... Ts,
      typename = std::enable_if_t<sizeof...(Ts) == arity>>
    static const std::tuple<Ts...>&
    search_pattern(const std::tuple<Ts...>& pattern)
    {
      return pattern;
    }

    template<
      size_t I,
      typename = std::enable_if_t<I<arity>> const
        std::tuple_element_t<I, Tuple>& value_index(const value_type& value)
    {
      return std::get<I>(value);
    }

    template<typename... Ts>
    static value_type value_construct(Ts&&... values)
    {
      return std::make_tuple(std::forward<Ts>(values)...);
    }
  };

  template<typename Tuple, typename Compare, size_t... Idx>
  struct RelationMeta<
    Tuple,
    Compare,
    std::index_sequence<Idx...>,
    std::enable_if_t<is_lattice<
      std::tuple_element_t<std::tuple_size_v<Tuple> - 1, Tuple>>::value>>
  {
    using key_type = std::tuple<std::tuple_element_t<Idx, Tuple>...>;
    using mapped_type =
      std::tuple_element_t<std::tuple_size_v<Tuple> - 1, Tuple>;
    using value_type = std::pair<key_type, mapped_type>;
    using container_type = std::map<key_type, mapped_type, Index<Compare>>;

    static constexpr size_t arity = std::tuple_size_v<Tuple>;

    template<
      typename... Ts,
      typename = std::enable_if_t<sizeof...(Ts) == arity>>
    static auto search_pattern(const std::tuple<Ts...>& pattern)
    {
      return std::make_tuple(std::get<Idx>(pattern)...);
    }

    template<
      size_t I,
      typename = std::enable_if_t<I<arity>> const
        std::tuple_element_t<I, Tuple>& value_index(const value_type& value)
    {
      if constexpr (I < arity - 1)
        return std::get<I>(value.first);
      else
        return value.second;
    }

    value_type static value_construct(
      std::tuple_element_t<Idx, Tuple>... key, mapped_type mapped)
    {
      return std::make_pair(std::make_tuple(key...), mapped);
    }
  };

  template<typename Tuple, typename Compare = std::less<>>
  struct Relation : public RelationMeta<Tuple, Compare>
  {
    using meta = RelationMeta<Tuple, Compare>;
    using tuple_type = Tuple;
    using index_type = Index<Compare>;

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

    template<typename B, typename T, typename U>
    static std::conditional_t<(std::is_placeholder_v<U>> 0), B, T>
    make_bound(const U& value)
    {
      if constexpr (std::is_placeholder_v<U>> 0)
        return B();
      else
        return T(value);
    }

    template<typename B, typename... Ts, size_t... Is>
    static auto
    make_bound(const std::tuple<Ts...>& values, std::index_sequence<Is...>)
    {
      return std::make_tuple(make_bound<B, std::tuple_element_t<Is, Tuple>>(
        std::get<Is>(values))...);
    }

    template<ExecutionMode mode = ExecutionMode::Stable, typename... Ts>
    auto search(const std::tuple<Ts...>& pattern) const
    {
      auto lower = meta::search_pattern(
        make_bound<lower_limit>(pattern, std::index_sequence_for<Ts...>()));
      auto upper = meta::search_pattern(
        make_bound<upper_limit>(pattern, std::index_sequence_for<Ts...>()));

      if constexpr (mode == ExecutionMode::Stable)
        return std::make_pair(
          stable_values.lower_bound(lower), stable_values.upper_bound(upper));
      else
        return std::make_pair(
          recent_values.lower_bound(lower), recent_values.upper_bound(upper));
    }

    template<typename... T>
    void emplace(T&&... values)
    {
      pending_values.emplace_back(
        meta::value_construct(std::forward<T>(values)...));
    }

    void add(Tuple key)
    {
      pending_values.push_back(std::apply(meta::value_construct, key));
    }

    template<
      typename... Ts,
      typename =
        std::enable_if_t<is_valid_selector<std::tuple<Ts...>, Tuple>::value>>
    Selector<Relation, Ts...> operator()(Ts... keys)
    {
      return Selector(*this, std::make_tuple(keys...));
    }

    Relation() = default;
    Relation(const Relation& other) = delete;

  private:
    typename meta::container_type stable_values;
    typename meta::container_type recent_values;

    std::vector<typename meta::value_type> pending_values;
  };

  template<
    typename Tuple,
    typename Compare,
    typename P,
    typename =
      std::enable_if_t<std::is_convertible_v<typename P::result_type, Tuple>>>
  void operator+=(Relation<Tuple, Compare>& relation, P&& producer)
  {
    std::forward<P>(producer).template execute<ExecutionMode::Delta>(
      [&](Tuple entry) { relation.add(entry); });
  }
}
