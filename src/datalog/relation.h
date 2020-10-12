#pragma once

#include "datalog/lattice.h"
#include "datalog/type_traits.h"

#include <iostream>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace datalog
{
  template<typename R, typename... Ts>
  struct Selector;

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
   * lower_limit and upper_limit are special marker types that respectively
   * compare as smaller and greater than any other value.
   *
   * They are used to searches inside ordered collections of tuples, where we
   * don't care about the particular value of some of the tuple elements.
   */
  struct lower_limit
  {};
  struct upper_limit
  {};

  struct AbstractRelation;
  struct Strata
  {
    enum class State
    {
      Empty,
      Initialization,
      InProgress,
      Done,
    };

    Strata() = default;
    // We don't allow copying nor moving a Strata, because the relations hold
    // references to it.
    Strata(const Strata&) = delete;
    Strata& operator=(const Strata&) = delete;

    template<typename Fn>
    void compute(Fn&& fn);

    void add(AbstractRelation* relation)
    {
      assert(state_ == State::Empty);
      relations.push_back(relation);
    }

    State state() const
    {
      return state_;
    }

  private:
    State state_ = State::Empty;
    std::vector<AbstractRelation*> relations;

    friend AbstractRelation;
  };

  /// Type-erased version of a Relation, used to be stored in a Strata.
  struct AbstractRelation
  {
    AbstractRelation(Strata& strata) : strata(strata)
    {
      strata.add(this);
    }
    virtual bool update() = 0;
    virtual ~AbstractRelation(){};

    // We don't allow copying nor moving an AbstractRelation, because the
    // Strata holds a pointer to it.
    AbstractRelation(const AbstractRelation&) = delete;
    AbstractRelation& operator=(const AbstractRelation&) = delete;

    Strata::State state()
    {
      return strata.state();
    }

  private:
    Strata& strata;
  };

  template<typename Fn>
  void Strata::compute(Fn&& fn)
  {
    assert(state_ == State::Empty);
    state_ = State::Initialization;
    fn();
    state_ = State::InProgress;
    while (true)
    {
      bool result = false;
      for (auto& it : relations)
      {
        result |= it->update();
      }

      if (!result)
        break;

      fn();
    }
    state_ = State::Done;
  }

  /**
   * Index performs comparisons of tuples in lexicographical order, with support
   * for lower_limit and upper_limit marker types.
   */
  template<typename Compare>
  struct Index
  {
    // Make the a comparator transparent, allowing lookups in containers to use
    // types that aren't the key's type.
    using is_transparent = void;

    template<typename T>
    static constexpr bool is_limit =
      (std::is_same_v<T, upper_limit> || std::is_same_v<T, lower_limit>);

    template<typename... Ts>
    static constexpr bool has_limits = ((is_limit<Ts>) || ...);

    /**
     * Compare two tuples in lexicographical order, taking into account any
     * lower_ or upper_limit.
     *
     * We only allow comparisons where at most one side mentions a limit.
     * Failing to do so could
     */
    template<
      typename... Ts,
      typename... Us,
      typename = std::enable_if_t<sizeof...(Ts) == sizeof...(Us)>,
      typename = std::enable_if_t<(!has_limits<Ts...> || !has_limits<Us...>)>>
    bool operator()(
      const std::tuple<Ts...>& left, const std::tuple<Us...>& right) const
    {
      return compare_tuples(left, right, std::index_sequence_for<Ts...>());
    }

  private:
    template<typename Ts, typename Us, size_t I, size_t... Is>
    bool compare_tuples(
      const Ts& left, const Us& right, std::index_sequence<I, Is...>) const
    {
      using T = std::tuple_element_t<I, Ts>;
      using U = std::tuple_element_t<I, Us>;

      if constexpr (
        std::is_same_v<T, lower_limit> || std::is_same_v<U, upper_limit>)
      {
        return true;
      }
      else if constexpr (
        std::is_same_v<T, upper_limit> || std::is_same_v<U, lower_limit>)
      {
        return false;
      }
      else
      {
        const T& l = std::get<I>(left);
        const U& r = std::get<I>(right);
        if (Compare()(l, r))
          return true;
        else if (Compare()(r, l))
          return false;
        else
          return compare_tuples(left, right, std::index_sequence<Is...>());
      }
    }

    template<typename Ts, typename Us>
    bool
    compare_tuples(const Ts& left, const Us& right, std::index_sequence<>) const
    {
      return false;
    }
  };

  /// The underlying storage of a Relation depends on whether a lattice
  /// component is present or not. If such an element is present, the relation
  /// is a map from its non-lattice part to its lattice element. Otherwise the
  /// relation is a set.
  ///
  /// We use two different specializations of `relation_meta` to pick the right
  /// types and operations for each case.

  template<
    typename Tuple,
    typename Compare = std::less<>,
    bool lattice = is_lattice_v<tuple_last_element_t<Tuple>>>
  struct RelationBase;

  template<typename Tuple, typename Compare>
  struct RelationBase<Tuple, Compare, false> : public AbstractRelation
  {
    using AbstractRelation::AbstractRelation;

    static_assert(forall<std::tuple_size_v<Tuple>>([](auto I) {
      return !is_lattice_v<std::tuple_element_t<I, Tuple>>;
    }));

    using tuple_type = Tuple;
    using key_type = Tuple;
    using value_type = Tuple;
    using container_type = std::set<Tuple, Index<Compare>>;

    void insert(Tuple key)
    {
      assert(
        state() == Strata::State::Initialization ||
        state() == Strata::State::InProgress);
      pending_values.push_back(key);
    }

    bool update() final
    {
      assert(
        state() == Strata::State::Initialization ||
        state() == Strata::State::InProgress);
      for (const auto& value : recent_values)
      {
        auto [it, inserted] = stable_values.insert(value);
        assert(inserted);
      }
      recent_values.clear();

      for (const auto& value : pending_values)
      {
        if (stable_values.count(value) == 0)
          recent_values.insert(value);
      }
      pending_values.clear();

      return !recent_values.empty();
    }

    template<ExecutionMode mode>
    const container_type& values() const
    {
      if constexpr (mode == ExecutionMode::Stable)
        return stable_values;
      else
        return recent_values;
    }

  private:
    container_type stable_values;
    container_type recent_values;
    std::vector<value_type> pending_values;
  };

  template<typename Tuple, typename Compare>
  struct RelationBase<Tuple, Compare, true> : public AbstractRelation
  {
    using AbstractRelation::AbstractRelation;

    static_assert(forall<std::tuple_size_v<Tuple> - 1>([](auto I) {
      return !is_lattice_v<std::tuple_element_t<I, Tuple>>;
    }));
    static_assert(is_lattice_v<tuple_last_element_t<Tuple>>);

    using tuple_type = Tuple;
    using key_type = tuple_slice_t<0, std::tuple_size_v<Tuple> - 1, Tuple>;
    using mapped_type = tuple_last_element_t<Tuple>;
    using value_type = std::pair<const key_type, mapped_type>;
    using container_type = std::map<key_type, mapped_type, Index<Compare>>;

    void insert(const Tuple& tuple)
    {
      key_type key = generate_tuple<std::tuple_size_v<Tuple> - 1>(
        [&](auto I) { return std::get<I>(tuple); });

      mapped_type mapped = std::get<std::tuple_size_v<Tuple> - 1>(tuple);

      pending_values.emplace_back(key, mapped);
    }

    bool update() final
    {
      for (const auto& value : recent_values)
      {
        auto [it, inserted] = stable_values.insert(value);
        if (!inserted)
        {
          assert(
            mapped_type::compare(it->second, value.second) ==
            partial_ordering::less);
          it->second = value.second;
        }
      }
      recent_values.clear();

      for (auto value : pending_values)
      {
        auto stable_it = stable_values.find(value.first);
        if (stable_it != stable_values.end())
        {
          auto cmp = mapped_type::compare(value.second, stable_it->second);
          if (
            cmp == partial_ordering::less ||
            cmp == partial_ordering::equivalent)
            continue;

          value.second = mapped_type::lub(value.second, stable_it->second);
        }

        auto [recent_it, inserted] = recent_values.insert(value);
        if (!inserted)
        {
          recent_it->second = mapped_type::lub(recent_it->second, value.second);
        }
      }
      pending_values.clear();

      return !recent_values.empty();
    }

    template<ExecutionMode mode>
    const container_type& values() const
    {
      if constexpr (mode == ExecutionMode::Stable)
        return stable_values;
      else
        return recent_values;
    }

  private:
    container_type stable_values;
    container_type recent_values;
    std::vector<value_type> pending_values;
  };

  template<typename Tuple, typename Compare = std::less<>>
  struct Relation : public RelationBase<Tuple, Compare>
  {
    using base = RelationBase<Tuple, Compare>;
    using base::base;

  public:
    using iterator = typename base::container_type::const_iterator;

    Relation() = default;
    Relation(const Relation& other) = delete;

    template<
      ExecutionMode mode,
      typename Query,
      typename = std::enable_if_t<is_valid_query_v<Query, Tuple>>>
    iterator lower_bound(const Query& query) const
    {
      auto search = make_search_bound<lower_limit>(query);
      return this->template values<mode>().lower_bound(search);
    }

    template<
      ExecutionMode mode,
      typename Query,
      typename = std::enable_if_t<is_valid_query_v<Query, Tuple>>>
    iterator upper_bound(const Query& query) const
    {
      auto search = make_search_bound<upper_limit>(query);
      return this->template values<mode>().upper_bound(search);
    }

    template<
      typename... Ts,
      typename =
        std::enable_if_t<is_valid_selector_v<std::tuple<Ts...>, Tuple>>>
    Selector<Relation, Ts...> operator()(Ts... pattern)
    {
      return Selector<Relation, Ts...>(*this, std::make_tuple(pattern...));
    }

  private:
    template<typename Limit, typename Query>
    static auto make_search_bound(const Query& query)
    {
      constexpr size_t N = std::tuple_size_v<typename base::key_type>;
      return generate_tuple<N>([&](auto I) {
        using P = std::tuple_element_t<I, Query>;

        if constexpr (std::is_placeholder_v<P> != 0)
          return Limit();
        else
          return std::get<I>(query);
      });
    }
  };
}
