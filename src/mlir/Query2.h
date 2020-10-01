#pragma once

// #include "llvm/ADT/ArrayRef.h"

namespace mlir::verona
{
#if 0
  template<typename... Ts>
  struct Environment
  {
    using tuple_type = std::tuple<Ts...>;

    /**
     * Create an empty environment. This constructor is only available if
     * Ts is empty.
     */
    template<
      bool enable = sizeof...(Ts) == 0,
      typename = std::enable_if_t<enable>>
    explicit Environment()
    {}

    explicit Environment(std::tuple<Ts...> values) : values(values) {}

    /**
     * Returns true if placeholder P has an associated value in the current
     * environment.
     */
    template<size_t P>
    static constexpr bool has_value()
    {
      if constexpr (P <= sizeof...(Ts))
        return std::is_placeholder_v<
                 std::tuple_element_t<P - 1, std::tuple<Ts...>>> == 0;
      else
        return false;
    }

    template<size_t P>
    using value_type = std::tuple_element_t<P - 1, std::tuple<Ts...>>;

    template<size_t P>
    value_type<P> get_value() const
    {
      return std::get<P - 1>(values);
    }

  public:
    template<typename... Us>
    auto unify(const Environment<Us...>& other)
    {
      constexpr size_t N = std::max(sizeof...(Ts), sizeof...(Us));
      return unify_impl(std::make_index_sequence<N>());
    }

  private:
    template<size_t... I, typename... Us>
    auto unify_impl(const Environment<Us...>& other)
    {
      return make_environment(std::make_tuple(unify_at<I + 1>(other)...));
    }

    template<size_t P, typename... Us>
    auto unify_at(const Environment<Us...>& other)
    {
      using Other = Environment<Us...>;

      if constexpr (has_value<P>() && Other::template has_value<P>())
      {
        static_assert(
          std::is_same_v<value_type<P>, Other::template value_type<P>>);
        const auto& left = get_value<P>();
        const auto& right = other.template get_value<P>();

        assert(left == right);
        return right;
      }
      else if constexpr (has_value<P>())
      {
        return get_value<P>();
      }
      else if constexpr (Environment<Us...>::template has_value<P>())
      {
        return other.template get_value<P>();
      }
      else
      {
        return placeholder<P>();
      }
    }

  public:
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
     * An environment is complete if there are no gaps in the placeholders it
     * maps. For instance { _1 => B, _2 => C } is complete, but
     * { _1 => B, _3 => C } is not.
     */
    static constexpr bool is_complete =
      ((std::is_placeholder_v<Ts> == 0) && ...);

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
    template<
      typename Fn,
      bool enable = is_complete,
      std::enable_if_t<enable, int> = 0>
    std::invoke_result_t<Fn&&, const Ts&...> apply(Fn&& fn) const
    {
      return std::apply(std::forward<Fn>(fn), values);
    }

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
        static_assert(std::is_same_v<decltype(new_value), decltype(old_value)>);
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

  template<typename E, typename Q, typename R>
  using extended_environment = decltype(std::declval<const E&>().extend(
    std::declval<const Q&>(), std::declval<const R&>()));

  template<typename S>
  using initial_environment = extended_environment<
    Environment<>,
    typename S::query_type,
    typename S::result_type>;
#endif

}
