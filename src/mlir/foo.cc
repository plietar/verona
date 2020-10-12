#include "datalog/join.h"
#include "datalog/lattice.h"
#include "datalog/relation.h"
#include "datalog/selector.h"

#include <iostream>
#include <variant>

using namespace datalog;
using namespace std::placeholders;

bool implies(bool left, bool right)
{
  return !left || right;
}

///        -- Top ---
///       /          \
///      /            \
///   Even            Odd
///  /    \          /    \
/// C(0)  C(2)    C(1)  C(3)
///  \      \      /      /
///   \      \    /      /
///    \---- Bottom ----/
///
struct Value
{
  bool is_even;
  bool is_odd;
  std::optional<uint32_t> constant_value;

  static Value Top()
  {
    return Value(true, true);
  }
  static Value Bottom()
  {
    return Value(false, false);
  }
  static Value Even()
  {
    return Value(true, false);
  }
  static Value Odd()
  {
    return Value(false, true);
  }

  static Value Constant(uint32_t x)
  {
    return Value(x);
  }

  bool is_constant() const
  {
    return constant_value.has_value();
  }

  Value(uint32_t x) : is_even(x % 2 == 0), is_odd(x % 2 == 1), constant_value(x)
  {}

  Value(bool is_even, bool is_odd)
  : is_even(is_even), is_odd(is_odd), constant_value(std::nullopt)
  {}

  friend std::ostream& operator<<(std::ostream& os, Value self)
  {
    if (self.is_constant())
      return os << "constant(" << *self.constant_value << ")";
    else if (self.is_even && self.is_odd)
      return os << "top";
    else if (self.is_even && !self.is_odd)
      return os << "even";
    else if (!self.is_even && self.is_odd)
      return os << "odd";
    else
      return os << "bottom";
  }
};

Value operator+(Value left, Value right)
{
  if (left.is_constant() && right.is_constant())
  {
    return Value(*left.constant_value + *right.constant_value);
  }
  else
  {
    bool is_even =
      (left.is_even && right.is_even) || (left.is_odd && right.is_odd);
    bool is_odd =
      (left.is_even && right.is_odd) || (left.is_odd && right.is_even);
    return Value(is_even, is_odd);
  }
}

bool operator==(Value left, Value right)
{
  return std::tie(left.constant_value, left.is_even, left.is_odd) ==
    std::tie(right.constant_value, right.is_even, right.is_odd);
}

template<>
struct datalog::lattice_traits<Value>
{
  static Value lub(Value left, Value right)
  {
    if (left == right)
      return left;
    else
      return Value(left.is_even || right.is_even, left.is_odd || right.is_odd);
  }

  static Value glb(Value left, Value right)
  {
    switch (compare(left, right))
    {
      case partial_ordering::equivalent:
        return left;
      case partial_ordering::less:
        return left;
      case partial_ordering::greater:
        return right;
      case partial_ordering::unordered:
        return Value::Bottom();
    }
  }

  static partial_ordering compare(Value left, Value right)
  {
    if (left == right)
      return partial_ordering::equivalent;
    else if (left.is_constant() && right.is_constant())
      return partial_ordering::unordered;
    else if (
      implies(left.is_odd, right.is_odd) &&
      implies(left.is_even, right.is_even))
      return partial_ordering::less;
    else if (
      implies(right.is_odd, left.is_odd) &&
      implies(right.is_even, left.is_even))
      return partial_ordering::greater;
    else
      return partial_ordering::unordered;
  }
};

int main()
{
  using L = lattice_traits<Value>;
  constexpr auto less = partial_ordering::less;
  constexpr auto greater = partial_ordering::greater;
  constexpr auto unordered = partial_ordering::unordered;

  assert(L::compare(Value::Bottom(), Value::Top()) == less);
  assert(L::compare(Value::Top(), Value::Bottom()) == greater);
  assert(L::compare(Value::Bottom(), Value::Constant(5)) == less);
  assert(L::compare(Value::Constant(4), Value::Constant(5)) == unordered);
  assert(L::compare(Value::Constant(4), Value::Top()) == less);
  assert(L::lub(Value::Constant(4), Value::Constant(6)) == Value::Even());
  assert(L::lub(Value::Constant(4), Value::Constant(5)) == Value::Top());
  assert(L::lub(Value::Constant(3), Value::Constant(5)) == Value::Odd());
  assert(L::lub(Value::Constant(3), Value::Even()) == Value::Top());
  assert(L::lub(Value::Constant(3), Value::Odd()) == Value::Odd());
  assert(L::lub(Value::Top(), Value::Top()) == Value::Top());
  assert(L::lub(Value::Bottom(), Value::Top()) == Value::Top());
  assert(L::lub(Value::Bottom(), Value::Bottom()) == Value::Bottom());
  assert(L::glb(Value::Constant(4), Value::Constant(6)) == Value::Bottom());
  assert(L::glb(Value::Constant(4), Value::Constant(5)) == Value::Bottom());
  assert(L::glb(Value::Constant(3), Value::Constant(5)) == Value::Bottom());
  assert(L::glb(Value::Constant(3), Value::Even()) == Value::Bottom());
  assert(L::glb(Value::Top(), Value::Top()) == Value::Top());
  assert(L::glb(Value::Bottom(), Value::Top()) == Value::Bottom());
  assert(L::glb(Value::Bottom(), Value::Bottom()) == Value::Bottom());

#if 0
  using ident = std::string_view;
  Strata program;
  Relation<std::tuple<ident, ident, ident>> stmt_add(program);
  Relation<std::tuple<ident, ident>> stmt_copy(program);
  Relation<std::tuple<ident, uint32_t>> stmt_constant(program);
  Relation<std::tuple<ident, ident, ident>> stmt_choice(program);
  program.compute([&]() {
    stmt_constant.insert({"a", 2});
    stmt_constant.insert({"b", 4});
    stmt_constant.insert({"c", 3});
    stmt_choice.insert({"d", "a", "b"});
    stmt_add.insert({"e", "d", "c"});
  });

  Strata analysis;
  Relation<std::tuple<ident, must_lattice<Value>>> values(analysis);
  analysis.compute([&]() {
    values(_1, _3) += stmt_copy(_1, _2) & values(_2, _3);
    values += stmt_choice(_1, _2, _3)
                .join(values(_2, _4))
                .join(values(_3, _5))
                .with([](ident x, ident, ident, Value y, Value z) {
                  return std::tuple(x, L::lub(y, z));
                });

    values += stmt_constant(_1, _2).with(
      [](ident x, uint32_t v) { return std::tuple(x, Value::Constant(v)); });

    values += stmt_add(_1, _2, _3)
                .join(values(_2, _4))
                .join(values(_3, _5))
                .with([](ident x, ident, ident, Value y, Value z) {
                  return std::tuple(x, y + z);
                });
  });

  for (auto [x, y] : values(_1, _2))
  {
    std::cout << x << ": " << *y << std::endl;
  }
#endif

  using instr = uint32_t;
  using var = std::string_view;
  using field = std::string_view;
  using addr = uint32_t;

  Strata program;
  Relation<std::tuple<var, addr>> stmt_new(program);
  Relation<std::tuple<var, var>> stmt_copy(program);
  Relation<std::tuple<var, var, field>> stmt_load(program);
  Relation<std::tuple<var, field, var>> stmt_store(program);

  Relation<std::tuple<var, addr>> var_points_to(program);
  Relation<std::tuple<addr, field, addr>> field_points_to(program);

  program.compute([&]() {
    var_points_to(_1, _2) += stmt_new(_1, _2) & stmt_new(_1, _2);
    var_points_to(_1, _3) += stmt_copy(_1, _2) & var_points_to(_2, _3);

    var_points_to(_1, _5) += stmt_load(_1, _2, _3) & var_points_to(_2, _4) &
      field_points_to(_4, _3, _5);

    field_points_to(_4, _2, _5) +=
      stmt_store(_1, _2, _3) & var_points_to(_1, _4) & var_points_to(_3, _5);

    stmt_new.insert({ "a", 0 });
    stmt_new.insert({ "b", 1 });
    stmt_store.insert({ "a", "f", "b" });
    stmt_copy.insert({ "b", "a" });
    stmt_load.insert({ "b", "b", "f" });
  });

  for (auto [x, y] : var_points_to(_1, _2))
  {
    std::cout << x << " --> " << y << std::endl;
  }

  for (auto [x, y, z] : field_points_to(_1, _2, _3))
  {
    std::cout << x << "." << y << " --> " << z << std::endl;
  }
}
