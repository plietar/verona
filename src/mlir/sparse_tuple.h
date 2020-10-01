#pragma once

namespace mlir::verona
{
  template<typename, typename>
  struct tuple_cons;

  template<typename T, typename... Ts>
  struct tuple_cons<T, std::tuple<Ts...>>
  {
    using type = std::tuple<T, Ts...>;
  };

  template<typename T, typename... Ts>
  using tuple_cons_t = tuple_cons<T, Ts...>;

  template<typename... Ts>
  struct sparse_tuple
  {
    template<size_t N = sizeof...(Ts)>
    struct storage_type;

    template<>
    struct storage_type<0>
    {};

    template<typename T, typename... T>
    struct storage_type<T, T...>
    {
      using type = std::conditional_t<std::is_void_v<T>, stro>;
    };
  };
};
