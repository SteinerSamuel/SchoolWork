package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of a goose.
 */
public final class Goose extends Waterfowl {
  /**
   * default constructor for the goose.
   *
   * @param name The name of the bird.
   */
  public Goose(String name) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.SEEDS, Food.VEGETATION)), "Lake");
  }

}
