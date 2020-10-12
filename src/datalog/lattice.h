#pragma once

#include <type_traits>

namespace datalog
{
  template<typename T>
  struct lattice_traits;

  enum class partial_ordering
  {
    less,
    equivalent,
    greater,
    unordered,
  };

  template<typename T, typename Traits = lattice_traits<T>>
  struct lattice
  {
    lattice(T value) : value(value) {}

    operator const T&() const
    {
      return value;
    }

    const T& operator*() const
    {
      return value;
    }

    T value;

    static lattice lub(const lattice& left, const lattice& right)
    {
      return lattice(Traits::lub(left.value, right.value));
    }

    static lattice glb(const lattice& left, const lattice& right)
    {
      return lattice(Traits::glb(left.value, right.value));
    }

    static partial_ordering compare(const lattice& left, const lattice& right)
    {
      return Traits::compare(left.value, right.value);
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

  struct unit
  {};

  template<>
  struct lattice_traits<unit>
  {
    static unit lub(unit, unit)
    {
      return unit{};
    }

    static unit gub(unit, unit)
    {
      return unit{};
    }
  };

  template<typename T, typename Traits>
  struct must_lattice_traits
  {
    static T lub(const T& left, const T& right)
    {
      return Traits::glb(left, right);
    }

    static T glb(const T& left, const T& right)
    {
      return Traits::lub(left, right);
    }

    static partial_ordering compare(const T& left, const T& right)
    {
      switch (Traits::compare(left, right))
      {
        case partial_ordering::less:
          return partial_ordering::greater;
        case partial_ordering::equivalent:
          return partial_ordering::equivalent;
        case partial_ordering::greater:
          return partial_ordering::less;
        case partial_ordering::unordered:
          return partial_ordering::unordered;
      }
    }
  };

  template<typename T, typename Traits = lattice_traits<T>>
  using must_lattice = lattice<T, must_lattice_traits<T, Traits>>;
};
