package bird;

import java.util.StringJoiner;

/**
 * An abstraction of the Water bird implementation which extends Bird.
 */
public abstract class WaterBirdAbstract extends BirdAbstract implements WaterBird {
  protected String bodyOfWater;

  @Override
  public String getBodyOfWater() {
    return bodyOfWater;
  }

  @Override
  public String toString() {
    String birdType = this.getClass().getName().split("\\.")[1];
    StringJoiner dietString = new StringJoiner(",");
    StringJoiner characteristicsString = new StringJoiner(",");
    for (Food f : dietPreference) {
      dietString.add(' ' + f.toString().toLowerCase().replace('_', ' '));
    }
    for (String s : characteristics) {
      characteristicsString.add(' ' + s);
    }
    return String.format("%s is a %s: %ss are %s extinct have %d wings and eat%s. "
                    + "They have the following characteristics:%s. %ss live near water %ss "
                    + "live near %s \n",
            name, birdType, birdType, (extinct) ? "" : "not", numOfWings, dietString.toString(),
            characteristicsString.toString(), birdType, birdType, bodyOfWater);
  }
}
