package bird;

import java.util.ArrayList;

/**
 * A representation for a single bird.
 */
public interface Bird {
  /**
   * Returns the extinction status of the bird.
   *
   * @return The extinction status of the bird.
   */
  boolean getExtinct();

  /**
   * Returns the name of the bird.
   *
   * @return The bird's name.
   */
  String getBirdName();

  /**
   * Returns the number of wings.
   *
   * @return The number of wings.
   */
  int getNumOfWings();

  /**
   * Returns a list of the food the bird will eat.
   *
   * @return A List of food the bird will eat.
   */
  ArrayList<Food> getDietPreference();

  /**
   * Returns a list of characterisitcs of the bird.
   *
   * @return A list of characteristics of the bird.
   */
  ArrayList<String> getCharacteristics();
}
