package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class which represents the flightless bird classification of bird.
 */
public class FlightlessBird extends BirdAbstract {
  private final ArrayList<String> baseCharacteristics = new ArrayList<>(
          Arrays.asList("Flightless"));

  /**
   * The default constructor for the Flightless Bird class.
   *
   * @param name    The name of the bird.
   * @param extinct Is the bird extinct.
   * @param wings   the number of wings the bird has.
   * @param diet    The diet the bird has as a list of Food.
   */
  public FlightlessBird(String name, boolean extinct, int wings, ArrayList<Food> diet) {
    this.name = name;
    this.extinct = extinct;
    this.numOfWings = wings;
    this.dietPreference = diet;
    this.characteristics = baseCharacteristics;
  }

  /**
   * A constructor used if a flightless bird has characteristics which are outside the base List.
   *
   * @param name                  The name of the bird.
   * @param extinct               Is the bird extinct.
   * @param wings                 the number of wings the bird has.
   * @param diet                  The diet the bird has as a list of Food.
   * @param extendCharacteristics A list of characteristics which extends the base.
   */
  public FlightlessBird(String name, boolean extinct, int wings, ArrayList<Food> diet,
                        ArrayList<String> extendCharacteristics) {
    this.name = name;
    this.extinct = extinct;
    this.numOfWings = wings;
    this.dietPreference = diet;
    extendCharacteristics.addAll(baseCharacteristics);
    this.characteristics = extendCharacteristics;
  }

}
