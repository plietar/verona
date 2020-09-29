#pragma once
#include "Environment.h"
#include "helpers.h"

#include <type_traits>

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
   * If the first element of the selector (Ph) is a placeholder, the first
   * element of the relation (T) can be anything.
   */
  template<typename Ph, typename T, typename... S, typename... C>
  struct is_valid_selector<
    std::tuple<Ph, S...>,
    std::tuple<T, C...>,
    std::enable_if_t<(std::is_placeholder_v<Ph>> 0)>>
  : is_valid_selector<std::tuple<S...>, std::tuple<C...>>
  {};

  /**
   * If the first element of the selector (T) is not a placeholder, it must be
   * the same as the first element of the relation.
   */
  template<typename T, typename... S, typename... C>
  struct is_valid_selector<
    std::tuple<T, S...>,
    std::tuple<T, C...>,
    std::enable_if_t<(std::is_placeholder_v<T> == 0)>>
  : is_valid_selector<std::tuple<S...>, std::tuple<C...>>
  {};

  template<typename... S>
  struct Join;
  template<typename Fn, typename... S>
  struct Producer;

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
    using relation_type = R;
    using query_type = std::tuple<Ts...>;
    using result_type = typename R::tuple_type;
    using environment_type = initial_environment_t<query_type, result_type>;

    static_assert(
      sizeof...(Ts) == std::tuple_size_v<typename R::tuple_type>,
      "The selector must have the same arity as the relation it refers to");
    static_assert(
      is_valid_selector<std::tuple<Ts...>, typename R::tuple_type>::value);

    R& relation;
    std::tuple<Ts...> values;

    Selector(R& relation, std::tuple<Ts...> values)
    : relation(relation), values(values)
    {}

    template<
      typename R2,
      typename... Ts2,
      typename = std::enable_if_t<can_unify_v<
        environment_type,
        typename Selector<R2, Ts2...>::environment_type>>>
    auto join(Selector<R2, Ts2...> other) const
    {
      return Join(*this, other);
    }

    template<
      typename R2,
      typename... Ts2,
      typename = std::enable_if_t<can_unify_v<
        environment_type,
        typename Selector<R2, Ts2...>::environment_type>>>
    auto operator&(Selector<R2, Ts2...> other) const
    {
      return Join(*this, other);
    }

    template<typename... S>
    void operator+=(const Join<S...>& join)
    {
      relation +=
        join.with([&](const auto& env) { return env.substitute(values); });
    }
  };

  template<typename... S>
  struct Join
  {
    static_assert(sizeof...(S) < 64);

    template<size_t N>
    struct result_type_impl;

    template<size_t N>
    struct result_type_impl
    {
      static_assert(N <= sizeof...(S));
      using selector = std::tuple_element_t<N - 1, std::tuple<S...>>;
      using type = unified_environment_t<
        typename result_type_impl<N - 1>::type,
        initial_environment_t<
          typename selector::query_type,
          typename selector::result_type>>;
    };

    template<>
    struct result_type_impl<0>
    {
      using type = Environment<>;
    };

    using result_type = typename result_type_impl<sizeof...(S)>::type;

    explicit Join(std::tuple<S...> selectors) : selectors(selectors) {}

    template<
      typename S1,
      typename S2,
      bool enable = sizeof...(S) == 2,
      typename = std::enable_if_t<enable>>
    explicit Join(S1 s1, S2 s2) : selectors{s1, s2}
    {}

    /**
     * Extend a join with another Selector.
     */
    template<typename R, typename... Ts>
    Join<S..., Selector<R, Ts...>> join(Selector<R, Ts...> other) const
    {
      return {std::tuple_cat(selectors, std::make_tuple(other))};
    }

    /**
     * Execute this join, invoking fn for each Environment this join produces.
     */
    template<ExecutionMode mode, typename Fn>
    void execute(Fn&& fn) const
    {
      if constexpr (mode == ExecutionMode::Stable)
        execute_impl<0, 0>(std::forward<Fn>(fn), empty_environment());
      else
        execute_delta(std::forward<Fn>(fn));
    }

    template<
      typename Fn,
      typename = std::enable_if_t<std::is_invocable_v<Fn, result_type>>>
    Producer<Fn, S...> with(Fn fn) const
    {
      return Producer<Fn, S...>{fn, *this};
    }

    template<
      typename Fn,
      typename = std::enable_if_t<
        result_type::is_complete &&
        is_applicable_v<Fn, typename result_type::tuple_type>>>
    auto with(Fn fn) const
    {
      return with(
        [=](const auto& env) { return std::apply(fn, env.values()); });
    }

  private:
    std::tuple<S...> selectors;

    template<uint64_t Recent = 1, typename Fn>
    void execute_delta(Fn&& fn) const
    {
      static_assert(Recent > 0);
      if constexpr (Recent < 1 << sizeof...(S))
      {
        execute_impl<0, Recent>(fn, empty_environment());
        execute_delta<Recent + 1>(fn);
      }
    }

    template<size_t I, uint64_t Recent, typename Fn, typename Env>
    void execute_impl(Fn&& fn, const Env& env) const
    {
      if constexpr (I == sizeof...(S))
      {
        std::forward<Fn>(fn)(env);
      }
      else
      {
        constexpr ExecutionMode mode = (Recent & (1 << I)) == 0 ?
          ExecutionMode::Stable :
          ExecutionMode::Delta;

        const auto& selector = std::get<I>(selectors);
        auto pattern = env.substitute(selector.values);

        auto [begin, end] = selector.relation.template search<mode>(pattern);
        for (auto it = begin; it != end; it++)
        {
          auto updated = unify(env, make_environment(selector.values, *it));
          execute_impl<I + 1, Recent>(fn, updated);
        }
      }
    }
  };

  template<typename S1, typename S2>
  Join(S1, S2)->Join<S1, S2>;

  template<typename Fn, typename... S>
  struct Producer
  {
    using result_type =
      std::invoke_result_t<Fn, typename Join<S...>::result_type>;

    Fn fn;
    Join<S...> join;

    template<ExecutionMode mode, typename Cb>
    void execute(Cb&& cb) const
    {
      join.template execute<mode>([&](const auto& env) { cb(fn(env)); });
    }
  };
}
