package bird;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class which represents the Shore Bird classificaiton.
 */
public class ShoreBird extends WaterBirdAbstract {
  private final ArrayList<String> baseCharacteristics = new ArrayList<>(
          Arrays.asList("Small", "Live near water"));

  /**
   * The default constructor for the Shore Bird class.
   *
   * @param name        The name of the bird
   * @param extinct     Is the bird extinct.
   * @param wings       the number of wings the bird has.
   * @param diet        The diet the bird has as a list of Food.
   * @param bodyOfWater The body of water the bird lives near.
   */
  public ShoreBird(String name, boolean extinct, int wings, ArrayList<Food> diet,
                   String bodyOfWater) {
    this.name = name;
    this.extinct = extinct;
    this.numOfWings = wings;
    this.dietPreference = diet;
    this.characteristics = baseCharacteristics;
    this.bodyOfWater = bodyOfWater;
  }

  /**
   * A constructor used if a Shore Bird has characteristics which are outside the base List.
   *
   * @param name                  The name of the bird.
   * @param extinct               Is the bird extinct.
   * @param wings                 the number of wings the bird has.
   * @param diet                  The diet the bird has as a list of Food.
   * @param bodyOfWater           The body of water the bird lives near by.
   * @param extendCharacteristics A list of characteristics which extends the base.
   */
  public ShoreBird(String name, boolean extinct, int wings, ArrayList<Food> diet,
                   String bodyOfWater, ArrayList<String> extendCharacteristics) {
    this.name = name;
    this.extinct = extinct;
    this.numOfWings = wings;
    this.dietPreference = diet;
    extendCharacteristics.addAll(baseCharacteristics);
    this.characteristics = extendCharacteristics;
    this.bodyOfWater = bodyOfWater;
  }

}