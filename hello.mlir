module attributes { vam.main_class = @Main, vam.main_method = #vam.selector<0> }
{
  func @"Hello.final"(%this: !vam.value) -> !vam.value {
    %msg = vam.literal "Finaliser {:#}\n"
    vam.print %msg, %this
    vam.drop %msg
    %ret = vam.literal unit
    vam.return %ret
  }

  func @"Hello.hello"(%this: !vam.value) -> !vam.value {
    %msg = vam.literal "Finaliser {:#}\n"
    vam.print %msg, %this
    vam.drop %msg
    %ret = vam.literal unit
    vam.return %ret
  }

  func @"Main.main"(%this: !vam.value) -> !vam.value {
    %x = vam.new_region @Hello
    %c = vam.new_cown @"cown[Hello]", %x

    vam.when cowns=(%c) behaviour=(%b) {
      %msg = vam.literal "In behaviour\n"
      %builtin = vam.load_descriptor @Builtin
      vam.call #vam.selector<2> (%builtin, %msg)
      vam.drop %msg
      vam.drop %builtin

      %ret = vam.literal unit
      vam.return %ret
    }
    vam.drop %c

    %ret = vam.literal unit
    vam.return %ret
  }

  func @"Builtin.print"(%this: !vam.value, %arg: !vam.value) -> !vam.value {
    vam.print %arg
    vam.drop %arg
    %ret = vam.literal unit
    vam.return %ret
  }

  vam.descriptor @"cown[Hello]" { }
  vam.descriptor @Hello attributes {  }
  // finaliser = @"Hello.final"
  {
    vam.field #vam.selector<0>
    vam.method #vam.selector<0> -> @"Hello.final"
    vam.method #vam.selector<1> -> @"Hello.hello"
  }

  vam.descriptor @Main
  {
    vam.method #vam.selector<0> -> @"Main.main"
  }

  vam.descriptor @Builtin  {
    vam.method #vam.selector<2> -> @"Builtin.print"
  }
}
