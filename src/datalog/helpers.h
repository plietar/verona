#pragma once

#include <tuple>

namespace datalog
{
  template<typename F, typename Tuple, typename = void>
  struct is_applicable : std::false_type
  {};

  template<typename F, typename... Ts>
  struct is_applicable<F, std::tuple<Ts...>> : std::is_invocable<F, Ts...>
  {};

  template<typename F, typename Tuple>
  static constexpr bool is_applicable_v = is_applicable<F, Tuple>::value;

  template<
    size_t... I,
    typename Fn,
    typename = std::enable_if_t<std::conjunction_v<
      std::is_invocable<Fn, std::integral_constant<size_t, I>>...>>>
  static constexpr std::tuple<
    std::invoke_result_t<Fn, std::integral_constant<size_t, I>>...>
  generate_tuple_impl(Fn&& fn, std::index_sequence<I...> indices)
  {
    return std::make_tuple(fn(std::integral_constant<std::size_t, I>())...);
  }

  template<size_t N, typename Fn>
  static constexpr auto generate_tuple(Fn&& fn)
  {
    return generate_tuple_impl(
      std::forward<Fn>(fn), std::make_index_sequence<N>());
  }

  template<
    size_t... I,
    typename Fn,
    typename = std::enable_if_t<std::conjunction_v<
      std::is_invocable_r<bool, Fn, std::integral_constant<size_t, I>>...>>>
  static constexpr bool forall_impl(Fn&& fn, std::index_sequence<I...> indices)
  {
    return (fn(std::integral_constant<size_t, I>()) && ...);
  }

  template<size_t N, typename Fn>
  static constexpr bool forall(Fn&& fn)
  {
    return forall_impl(std::forward<Fn>(fn), std::make_index_sequence<N>());
  }

  template<
    size_t... I,
    typename Fn,
    typename = std::enable_if_t<std::conjunction_v<
      std::is_invocable_r<bool, Fn, std::integral_constant<size_t, I>>...>>>
  static constexpr bool exists_impl(Fn&& fn, std::index_sequence<I...> indices)
  {
    return (fn(std::integral_constant<size_t, I>()) && ...);
  }

  template<size_t N, typename Fn>
  static constexpr bool exists(Fn&& fn)
  {
    return exists_impl(std::forward<Fn>(fn), std::make_index_sequence<N>());
  }

  /// https://wg21.link/P1830R1
  template<bool value, typename... Args>
  static constexpr bool dependent_bool_value = value;

  template<typename... Args>
  static constexpr bool dependent_false = dependent_bool_value<false, Args...>;

  template<typename T1, typename T2>
  struct integer_sequence_cat;

  template<typename T, size_t... I, size_t... J>
  struct integer_sequence_cat<
    std::integer_sequence<T, I...>,
    std::integer_sequence<T, J...>>
  {
    using type = std::integer_sequence<T, I..., J...>;
  };
  template<typename T1, typename T2>
  using integer_sequence_cat_t = typename integer_sequence_cat<T1, T2>::type;

  template<typename T, T I, typename T2>
  using integer_sequence_cons_t =
    integer_sequence_cat_t<std::integer_sequence<T, I>, T2>;

  template<size_t I, typename T2>
  using index_sequence_cons_t = integer_sequence_cons_t<size_t, I, T2>;

  template<typename... Ts>
  struct has_trailing_void;

  template<>
  struct has_trailing_void<> : std::false_type
  {};

  template<typename T, typename... Ts>
  struct has_trailing_void<T, Ts...>
  {
    static constexpr size_t N = 1 + sizeof...(Ts);
    using last_element = std::tuple_element_t<N - 1, std::tuple<T, Ts...>>;
    static constexpr bool value = std::is_void_v<last_element>;
  };

  template<typename... Ts>
  static constexpr bool has_trailing_void_v = has_trailing_void<Ts...>::value;

  template<typename>
  struct is_tuple : std::false_type
  {};

  template<typename... T>
  struct is_tuple<std::tuple<T...>> : std::true_type
  {};

  template<
    size_t Start,
    size_t End,
    typename Tuple,
    typename Idx = std::make_index_sequence<End - Start>>
  struct tuple_slice;

  template<size_t Start, size_t End, typename Tuple, size_t... I>
  struct tuple_slice<Start, End, Tuple, std::index_sequence<I...>>
  {
    using type = std::tuple<std::tuple_element_t<I + Start, Tuple>...>;
  };

  template<size_t Start, size_t End, typename Tuple>
  using tuple_slice_t = typename tuple_slice<Start, End, Tuple>::type;

  template<typename Tuple>
  using tuple_last_element_t =
    std::tuple_element_t<std::tuple_size_v<Tuple> - 1, Tuple>;
}
