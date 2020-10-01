
struct Unit
{};
struct UnitLattice
{
  using value_type = Unit;
  Unit top()
  {
    return Unit();
  }
  Unit bottom()
  {
    return Unit();
  }
  Unit meet(Unit a, Unit b)
  {
    return Unit();
  }
  Unit join(Unit a, Unit b)
  {
    return Unit();
  }
  bool compare(Unit a, Unit b)
  {
    return true;
  }
};

template<typename Key, typename ValueLattice>
struct Relation
{
  using key_type = Key;
  using value_type = typename ValueLattice::value_type;

  std::map<Key, value_type> values;
};
