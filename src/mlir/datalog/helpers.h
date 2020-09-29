#pragma once

#include <tuple>

namespace mlir::verona
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
}
