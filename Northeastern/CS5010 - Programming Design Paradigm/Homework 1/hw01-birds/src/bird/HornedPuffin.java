package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of a horned puffin.
 */
public final class HornedPuffin extends ShoreBird {
  /**
   * default constructor for the horned puffin.
   *
   * @param name The name of the bird.
   */
  public HornedPuffin(String name) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.SEEDS, Food.VEGETATION)), "Ocean");
  }

}
