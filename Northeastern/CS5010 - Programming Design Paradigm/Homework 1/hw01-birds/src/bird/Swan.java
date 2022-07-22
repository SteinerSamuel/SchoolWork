package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of a swan.
 */
public final class Swan extends Waterfowl {
  /**
   * default constructor for the swan.
   *
   * @param name The name of the bird.
   */
  public Swan(String name) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.SEEDS, Food.VEGETATION)), "Lake");
  }

}
