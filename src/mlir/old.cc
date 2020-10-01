struct FactSet
{
  QueryEngine engine;
  Relation<Alias, 0, ValueCmp> aliases;

  BaseFacts() : aliases(engine) {}

  void print(llvm::raw_ostream& os, AsmState& state) const
  {
    aliases.print(os, state, "  ");
  }

  void process(MLIRContext* ctx, const DominanceInfo& dominance)
  {
    Type sendableType = getSendable(ctx);
    Type mutableType = getMut(ctx);

    codefined.from_cross(
      defined,
      defined,
      [&](const auto& r1, const auto& r2) -> std::optional<CoDefined> {
        Block* b1 = r1.value().getParentBlock();
        Block* b2 = r2.value().getParentBlock();
        if (dominance.dominates(b1, b2) || dominance.dominates(b2, b1))
          return CoDefined(r1.value(), r2.value());
        else
          return std::nullopt;
      });

    // distinct(x, y) :- codefined(x, y), x != y
    distinct.from_map(
      codefined, [&](const CoDefined& r) -> std::optional<Distinct> {
        if (r.left() != r.right())
          return Distinct(r.left(), r.right());
        else
          return std::nullopt;
      });

    // sendable(x) :-
    //   defined(x : T),
    //   T <: iso | imm.
    sendable.from_map(
      defined, [&](const Defined& r) -> std::optional<Sendable> {
        if (isSubtype(normalizeType(r.value().getType()), sendableType))
          return Sendable(r.value());
        else
          return std::nullopt;
      });

    // mutable(x) :-
    //   defined(x : T),
    //   T <: mut
    mutable_.from_map(defined, [&](const Defined& r) -> std::optional<Mutable> {
      if (isSubtype(normalizeType(r.value().getType()), mutableType))
        return Mutable(r.value());
      else
        return std::nullopt;
    });
  }
};

/*
struct BaseFacts
{
  QueryEngine engine;
  Relation<Defined, 0, ValueCmp> defined;
  Relation<CoDefined, 0, ValueCmp> codefined;
  Relation<Sendable, 0, ValueCmp> sendable;
  Relation<Mutable, 0, ValueCmp> mutable_;
  Relation<Distinct, 0, ValueCmp> distinct;

  BaseFacts()
  : defined(engine),
    codefined(engine),
    sendable(engine),
    mutable_(engine),
    distinct(engine)
  {}

  void initialize(FuncOp op)
  {
    for (Block& block : op)
    {
      add_definitions(block);
    }

    DominanceInfo dominance(op);
    while (engine.iterate())
    {
      process(op.getContext(), dominance);
    }
  }

  void add_definitions(Block& block)
  {
    for (Value v : block.getArguments())
    {
      if (isaVeronaType(v.getType()))
        defined.add(Defined(v));
    }

    for (Operation& op : block)
    {
      for (Value v : op.getResults())
      {
        if (isaVeronaType(v.getType()))
          defined.add(Defined(v));
      }
    }
  }

  void print(llvm::raw_ostream& os, AsmState& state) const
  {
    defined.print(os, state, "  ");
    distinct.print(os, state, "  ");
  }

  void process(MLIRContext* ctx, const DominanceInfo& dominance)
  {
    Type sendableType = getSendable(ctx);
    Type mutableType = getMut(ctx);

    codefined.from_cross(
      defined,
      defined,
      [&](const auto& r1, const auto& r2) -> std::optional<CoDefined> {
        Block* b1 = r1.value().getParentBlock();
        Block* b2 = r2.value().getParentBlock();
        if (dominance.dominates(b1, b2) || dominance.dominates(b2, b1))
          return CoDefined(r1.value(), r2.value());
        else
          return std::nullopt;
      });

    // distinct(x, y) :- codefined(x, y), x != y
    distinct.from_map(
      codefined, [&](const CoDefined& r) -> std::optional<Distinct> {
        if (r.left() != r.right())
          return Distinct(r.left(), r.right());
        else
          return std::nullopt;
      });

    // sendable(x) :-
    //   defined(x : T),
    //   T <: iso | imm.
    sendable.from_map(
      defined, [&](const Defined& r) -> std::optional<Sendable> {
        if (isSubtype(normalizeType(r.value().getType()), sendableType))
          return Sendable(r.value());
        else
          return std::nullopt;
      });

    // mutable(x) :-
    //   defined(x : T),
    //   T <: mut
    mutable_.from_map(
      defined, [&](const Defined& r) -> std::optional<Mutable> {
        if (isSubtype(normalizeType(r.value().getType()), mutableType))
          return Mutable(r.value());
        else
          return std::nullopt;
      });
  }
};
*/

struct FactSet
{
  QueryEngine engine;
  Relation<Alias, 0, ValueCmp> aliases;
  Relation<In, 0, ValueCmp> ins;
  Relation<From, 0, ValueCmp> froms;

  Relation<Alias, 1, ValueCmp> aliases_right;
  Relation<In, 1, ValueCmp> ins_right;
  Relation<From, 1, ValueCmp> froms_right;

  FactSet()
  : aliases(engine),
    ins(engine),
    froms(engine),
    aliases_right(engine),
    ins_right(engine),
    froms_right(engine)
  {}

  void add(Alias fact)
  {
    aliases.add(fact);
  }
  void add(In fact)
  {
    ins.add(fact);
  }
  void add(From fact)
  {
    froms.add(fact);
  }

  bool contains(const Alias& fact) const
  {
    return aliases.contains(fact);
  }
  bool contains(const In& fact) const
  {
    return ins.contains(fact);
  }
  bool contains(const From& fact) const
  {
    return froms.contains(fact);
  }

  template<typename T>
  LogicalResult require_fact(Operation* op, const T& fact) const
  {
    if (contains(fact))
      return success();

    AsmState state(op->getParentOfType<ModuleOp>());
    std::string fact_str;
    llvm::raw_string_ostream os(fact_str);
    fact.print(os, state);

    return op->emitError("Could not prove fact '") << os.str() << "'";
  }

  void initialize(const BaseFacts& base, FuncOp op, FactSet* prev = nullptr)
  {
    for (Block& block : op)
    {
      for (Operation& op : block)
      {
        if (auto iface = llvm::dyn_cast<RegionCheckInterface>(op))
        {
          iface.add_facts(*this);
        }
      }
    }

    for (Block& block : op)
    {
      for (auto it = block.pred_begin(); it != block.pred_end(); it++)
      {
        for (BlockArgument arg : block.getArguments())
        {
          Value param = getPhiParameter(arg, it);

          if (prev)
          {
            aliases.from_select(prev->aliases_right, param, [&](Alias r) {
              assert(r.right() == param);
              return Alias(r.left(), arg);
            });
          }
        }
      }
    }

    while (engine.iterate())
    {
      process(base);
    }
  }

  void print(llvm::raw_ostream& os, AsmState& state) const
  {
    aliases.print(os, state, "  ");
    // ins.print(os, state, "  ");
    // froms.print(os, state, "  ");
  }

  void process(const BaseFacts& base)
  {
    // Re-index the aliases relation with the second variable.
    aliases_right.from_copy(aliases);

    // alias(x, x) :-
    //   defined(x).
    aliases.from_map(base.defined, [](const auto& r) -> Alias {
      return Alias(r.value(), r.value());
    });

    // alias(x, y) :-
    //   alias(y, x).
    aliases.from_map(aliases, [](const auto& r) -> Alias {
      return Alias(r.right(), r.left());
    });

    // alias(x, z) :-
    //   alias(x, y),
    //   alias(y, z).
    aliases.from_join(
      aliases_right, aliases, [](const auto& r1, const auto& r2) -> Alias {
        assert(r1.right() == r2.left());
        return Alias(r1.left(), r2.right());
      });

    /*
     alias(x, y, 0) :- codefined(x, y).
     alias(x, y, n) :- "x = copy(y)".
     alias(x, y, n) :- alias(y, x, n).
     alias(x, y, n) :- alias(x, z, n), alias(z, y, n).
     alias(x, y, n) :- "x = phi(zs)", ∀z ∈ zs. alias(z, y, n - 1).

     alias(x, y) :- ∀n. alias(x, y, _)
     */
  }
};

void RegionCheckerPass::runOnFunction()
{
  DominanceInfo dominance(op);

  BaseFacts base;
  base.initialize(getOperation());

  FactSet facts;
  facts.initialize(base, getOperation());

  // RegionChecker ck(getOperation());
  // ck.run();

  // AsmState state(getOperation());
  // ck.facts.print(llvm::errs(), state);

  /*
  for (Operation& op : bb)
  {
    if (auto iface = llvm::dyn_cast<RegionCheckInterface>(op))
    {
      if (failed(iface.check_facts(ck)))
      {
        signalPassFailure();
        return;
      }
    }
  }
  */
}

void CopyOp::add_facts(FactSet& facts)
{
  facts.add(Alias(output(), input()));
}
LogicalResult CopyOp::check_facts(const FactSet& facts)
{
  return success();
}

void ViewOp::add_facts(FactSet& facts)
{
  facts.add(Alias(output(), input()));
}
LogicalResult ViewOp::check_facts(const FactSet& facts)
{
  return success();
}

void FieldReadOp::add_facts(FactSet& facts)
{
  facts.add(In(output(), origin(), getFieldType()));
}
LogicalResult FieldReadOp::check_facts(const FactSet& facts)
{
  return success();
}

void FieldWriteOp::add_facts(FactSet& facts)
{
  facts.add(From(output(), origin()));
}
LogicalResult FieldWriteOp::check_facts(const FactSet& facts)
{
  return facts.require_fact(*this, From(value(), output()));
}
}

/*
      ins_right.from_copy(ins);
      froms_right.from_copy(froms);

      // in(x, y, {}) :-
      //   alias(x, y).
      ins.from_map(aliases, [](const auto& r) -> In {
        return In(r.left(), r.right(), SmallVector<Type, 0>());
      });

      // in(x, z, T1 ++ T2) :-
      //   in(x, y, T1),
      //   in(y, z, T2).
      ins.from_join(
        ins_right,
        ins,
        [&](const auto& r1, const auto& r2) -> std::optional<In> {
          assert(r1.right() == r2.left());
          if (!base.codefined.contains(CoDefined(r1.left(), r2.right())))
            return std::nullopt;

          SmallVector<Type, 0> types;
          llvm::copy(r1.types(), std::back_inserter(types));
          llvm::copy(r2.types(), std::back_inserter(types));
          return In(r1.left(), r2.right(), types);
        });

      // from(x, y, {}) :-
      //   in(x, y, T).
      froms.from_map(
        ins, [](const In& r) -> From { return From(r.left(), r.right()); });

      // from(x, y, {}) :-
      //   sendable(x),
      //   defined(y).
      froms.from_cross(
        base.sendable,
        base.defined,
        [&](const Sendable& r1, const Defined& r2) -> std::optional<From> {
          if (!base.codefined.contains(CoDefined(r1.value(), r2.value())))
            return std::nullopt;

          return From(r1.value(), r2.value());
        });

      // from(x, z, T1 ++ T2) :-
      //   from(x, y, T1),
      //   in(y, z, T2).
      froms.from_join(
        froms_right,
        ins,
        [&](const From& r1, const In& r2) -> std::optional<From> {
          assert(r1.right() == r2.left());
          if (!base.codefined.contains(CoDefined(r1.left(), r2.right())))
            return std::nullopt;

          SmallVector<Type, 0> types;
          llvm::copy(r1.types(), std::back_inserter(types));
          llvm::copy(r2.types(), std::back_inserter(types));

          return From(r1.left(), r2.right(), types);
        });

      // in(x, y, T) :-
      //   from(x, y, T),
      //   mutable(x).
      ins.from_join(
        froms,
        base.mutable_,
        [&](const From& r1, const Mutable& r2) -> std::optional<In> {
          assert(r1.left() == r2.value());
          if (!base.codefined.contains(CoDefined(r1.left(), r2.value())))
            return std::nullopt;

          SmallVector<Type, 0> types;
          llvm::copy(r1.types(), std::back_inserter(types));
          return In(r1.left(), r1.right(), types);
        });
*/

/*
      ins_right.from_copy(ins);
      froms_right.from_copy(froms);

      // in(x, y, {}) :-
      //   alias(x, y).
      ins.from_map(aliases, [](const auto& r) -> In {
        return In(r.left(), r.right(), SmallVector<Type, 0>());
      });

      // in(x, y, {}) :-
      //   alias(x, y).
      ins.from_map(aliases, [](const auto& r) -> In {
        return In(r.left, r.right, {});
      });

      // in(x, z, T1 ++ T2) :-
      //   in(x, y, T1),
      //   in(y, z, T2).
      ins.from_join(
        ins_right,
        ins,
        [&](const auto& r1, const auto& r2) -> std::optional<In> {
          assert(r1.right() == r2.left());
          if (!base.codefined.contains(CoDefined(r1.left(), r2.right())))
            return std::nullopt;

          SmallVector<Type, 0> types;
          llvm::copy(r1.types(), std::back_inserter(types));
          llvm::copy(r2.types(), std::back_inserter(types));
          return In(r1.left(), r2.right(), types);
        });

      // from(x, y, {}) :-
      //   in(x, y, T).
      froms.from_map(
        ins, [](const In& r) -> From { return From(r.left(), r.right()); });

      // from(x, y, {}) :-
      //   sendable(x),
      //   defined(y).
      froms.from_cross(
        base.sendable,
        base.defined,
        [&](const Sendable& r1, const Defined& r2) -> std::optional<From> {
          if (!base.codefined.contains(CoDefined(r1.value(), r2.value())))
            return std::nullopt;

          return From(r1.value(), r2.value());
        });

      // from(x, z, T1 ++ T2) :-
      //   from(x, y, T1),
      //   in(y, z, T2).
      froms.from_join(
        froms_right,
        ins,
        [&](const From& r1, const In& r2) -> std::optional<From> {
          assert(r1.right() == r2.left());
          if (!base.codefined.contains(CoDefined(r1.left(), r2.right())))
            return std::nullopt;

          SmallVector<Type, 0> types;
          llvm::copy(r1.types(), std::back_inserter(types));
          llvm::copy(r2.types(), std::back_inserter(types));

          return From(r1.left(), r2.right(), types);
        });

      // in(x, y, T) :-
      //   from(x, y, T),
      //   mutable(x).
      ins.from_join(
        froms,
        base.mutable_,
        [&](const From& r1, const Mutable& r2) -> std::optional<In> {
          assert(r1.left() == r2.value());
          if (!base.codefined.contains(CoDefined(r1.left(), r2.value())))
            return std::nullopt;

          SmallVector<Type, 0> types;
          llvm::copy(r1.types(), std::back_inserter(types));
          return In(r1.left(), r1.right(), types);
        });
*/

/*

   alias(x, y) => ∀z. may_flow_into(z, x) -> may_flow_into(z, y)

 */
/*
  func @foo(%b : i1, %x: !D) {
     cond_br %b, ^bb1, ^bb2

   ^bb1:
     %y0 = verona.copy %x : !D -> !D
     br ^end(%y0 : !D)

   ^bb2:
     %y1 = verona.view %x : !D -> !D
     br ^end(%y1 : !D)

   ^end(%y: !D):
     return
  }
  */
