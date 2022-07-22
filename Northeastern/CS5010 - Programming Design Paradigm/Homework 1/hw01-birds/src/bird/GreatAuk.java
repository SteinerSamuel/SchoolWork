package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of a great auk.
 */
public final class GreatAuk extends ShoreBird {
  /**
   * default constructor for the great auk.
   *
   * @param name The name of the bird.
   */
  public GreatAuk(String name) {
    super(name, true, 2, new ArrayList<>(Arrays.asList(Food.SEEDS, Food.VEGETATION)), "Ocean");
  }

}
