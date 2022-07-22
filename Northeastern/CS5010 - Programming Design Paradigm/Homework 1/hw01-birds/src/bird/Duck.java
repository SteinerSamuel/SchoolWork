package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of a duck.
 */
public final class Duck extends Waterfowl {
  /**
   * default constructor for the duck.
   *
   * @param name The name of the bird.
   */
  public Duck(String name) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.SEEDS, Food.VEGETATION)), "Lake");
  }

}
