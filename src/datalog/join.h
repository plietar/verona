#pragma once
#include "datalog/helpers.h"
#include "datalog/selector.h"

#include <type_traits>

namespace datalog
{
  template<typename Fn, typename... S>
  struct Producer;

  template<typename E, typename... S>
  struct join_result;

  template<typename E>
  struct join_result<E>
  {
    using type = E;
  };

  template<typename E, typename S, typename... Tail>
  struct join_result<E, S, Tail...>
  {
    using result = typename rebind_selector_t<E, S>::iterator::value_type;
    using unified = unified_environment_t<E, result>;
    using type = typename join_result<unified, Tail...>::type;
  };

  template<typename... S>
  struct Join
  {
    static_assert(sizeof...(S) < 64);

    explicit Join(S... selectors) : selectors(selectors...) {}

    using result_type = typename join_result<Environment<>, S...>::type;

    /**
     * Extend a join with another Selector.
     */
    template<
      typename R,
      typename... Ts,
      typename = std::enable_if_t<is_valid_query_v<
        substitution_t<result_type, std::tuple<Ts...>, SubstituteLattice::No>,
        typename R::tuple_type>>>
    Join<S..., Selector<R, Ts...>> join(Selector<R, Ts...> other) const
    {
      return std::apply(
        [&](auto... selectors) {
          return Join<S..., Selector<R, Ts...>>(selectors..., other);
        },
        selectors);
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
        result_type::is_contiguous &&
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

        auto selector = std::get<I>(selectors).rebind(env);
        for (auto it = selector.template begin<mode>(); it != selector.end();
             ++it)
        {
          execute_impl<I + 1, Recent>(fn, unify(env, *it));
        }
      }
    }
  };

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

  template<
    typename Tuple,
    typename Compare,
    typename P,
    typename =
      std::enable_if_t<std::is_convertible_v<typename P::result_type, Tuple>>>
  void operator+=(Relation<Tuple, Compare>& relation, P&& producer)
  {
    if (relation.state() == Strata::State::Initialization)
      std::forward<P>(producer).template execute<ExecutionMode::Stable>(
        [&](Tuple entry) { relation.insert(entry); });
    else
      std::forward<P>(producer).template execute<ExecutionMode::Delta>(
        [&](Tuple entry) { relation.insert(entry); });
  }

  template<typename R1, typename R2, typename... Ts1, typename... Ts2>
  auto operator&(Selector<R1, Ts1...> left, Selector<R2, Ts2...> right)
  {
    return left.join(right);
  }

  template<typename... S, typename R, typename... Ts>
  auto operator&(const Join<S...>& left, const Selector<R, Ts...>& right)
  {
    return left.join(right);
  }
}
