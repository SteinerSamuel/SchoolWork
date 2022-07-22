package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A Class representation of a hawk.
 */
public final class Hawk extends BirdOfPray {
  /**
   * The default constructor of the Hawk class.
   *
   * @param name The name of the bird.
   */
  public Hawk(String name) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.SMALL_MAMMALS, Food.FISH,
            Food.OTHER_BIRDS)));
  }

}
