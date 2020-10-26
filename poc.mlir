module
{
  //
  // class Array[X] {
  //   create(): Array[X]
  //   push_back(self: Array[X], X): Unit
  // }
  //
  verona.class @Array [%X] {
    %Self = verona.class_type @Array[%X]
    verona.method @create {
      [] {
        %sig = verona.signature (%X) -> %Self
        verona.yield %sig
      }
      []() {
        verona.abort
      }
    }

    verona.method @push_back {
      [] {
        %unit = verona.unit_type
        %sig = verona.signature (%Self, %X) -> %unit
        verona.yield %sig
      }

      [](%self, %value) {
        verona.typecheck %self : %Self
        verona.typecheck %value : %X
        verona.abort
      }
    }
  }

  //
  // class Main {
  //   hello[X](value: X): Array[X] {
  //     let result = Array[X].create();
  //     result.push_back(value);
  //     return result
  //   }
  // }
  //
  verona.class @Main [] {
    verona.method @hello {
      [%X] {
        %Array = verona.class_type @Array[%X]
        %sig = verona.signature (%X) -> %Array
        verona.yield %sig
      }
      [%X](%value) {
        verona.typecheck %value : %X
        %Array = verona.class_type @Array[%X]
        %result = verona.static_call %Array "create" ()
        verona.typecheck %result : %Array
        verona.call %result "push_back" (%value)
        verona.return %result
      }
    }
  }
}
