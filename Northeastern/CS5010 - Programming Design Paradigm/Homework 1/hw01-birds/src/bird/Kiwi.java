package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of a kiwi.
 */
public final class Kiwi extends FlightlessBird {
  /**
   * The default constructor of the Kiwi class.
   */
  public Kiwi(String name) {
    super(name, true, 0, new ArrayList<>(Arrays.asList(Food.BUDS, Food.VEGETATION)),
            new ArrayList<>((Arrays.asList("Long beak", "Flexible beak"))));
  }

}
