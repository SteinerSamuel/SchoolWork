package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representation of an eagle.
 */
public final class Eagle extends BirdOfPray {
  /**
   * The default constructor of the Eagle class.
   *
   * @param name The name of the bird.
   */
  public Eagle(String name) {
    super(name, false, 2, new ArrayList<>(Arrays.asList(Food.OTHER_BIRDS, Food.FISH)));
  }

}
