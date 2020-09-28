#include "Query2.h"

#include <iostream>

using namespace mlir::verona;
using namespace std::placeholders;

enum class Colour
{
  Red,
  Blue
};
std::ostream& operator<<(std::ostream& os, Colour c)
{
  switch (c)
  {
    case Colour::Red:
      return os << "red";
    case Colour::Blue:
      return os << "blue";
  }
}

auto operator"" _w(long double);

int main()
{
  IndexedLattice<std::tuple<int, int>, Index> edges;
  edges.add({0, 1});
  edges.add({1, 2});
  edges.add({2, 3});
  edges.add({4, 5});

  {
    auto x = _1;
    auto y = _2;
    auto z = _3;

    do
    {
      edges(x, z) += edges(x, y) & edges(y, z);
    } while (edges.iterate());
  }

  /*
   */

  auto [begin, end] = edges.search_stable(std::make_tuple(_1, _1));
  for (auto it = begin; it != end; it++)
  {
    auto [x, y] = *it;
    std::cout << x << " " << y << std::endl;
  }
}
