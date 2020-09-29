#pragma once

#include <type_traits>

namespace mlir::verona
{
  template<typename T>
  struct lattice_traits;

  template<typename T, typename Traits = lattice_traits<T>>
  struct lattice
  {
    explicit lattice(T value) : value(value) {}
    T value;

    static lattice<T> lub(const lattice<T>& left, const lattice<T>& right)
    {
      return lattice<T>(Traits::lub(left.value, right.value));
    }

    static lattice<T> gub(const lattice<T>& left, const lattice<T>& right)
    {
      return lattice<T>(Traits::gub(left.value, right.value));
    }
  };

  template<typename T>
  struct is_lattice : std::false_type
  {};
  template<typename T, typename Traits>
  struct is_lattice<lattice<T, Traits>> : std::true_type
  {};
  template<typename T>
  static constexpr bool is_lattice_v = is_lattice<T>::value;
};
