#include "datalog/Environment.h"
#include "datalog/Join.h"
#include "datalog/Lattice.h"
#include "datalog/Relation.h"
#include "datalog/Selector.h"

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

template<>
struct mlir::verona::lattice_traits<int>
{
  static int lub(int left, int right)
  {
    return left & right;
  }
  static int gub(int left, int right)
  {
    return left | right;
  }
};

int main()
{
  Relation<std::tuple<int, char>> edges;
  /*
  edges.insert({0, 1});
  edges.insert({1, 2});
  edges.insert({2, 3});
  edges.insert({4, 5});

  while (edges.iterate())
  {
    edges(_1, _3) += edges(_1, _2) & edges(_2, _3);
  }
  */

  edges(_1, _2) & edges(_1, _1);

  /*
  Join(edges(_1, _2), edges(_2, _3))
  .execute<ExecutionMode::Stable>([&](const Environment<int, int, int>& e) {
    std::cout << e.get<0>() << " " << e.get<1>() << " " << e.get<2>()
              << std::endl;
  });
  */
}
