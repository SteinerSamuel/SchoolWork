package bird;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringJoiner;

/**
 * An abstract class of the Bird interface.
 */
public abstract class BirdAbstract implements Bird {
  protected String name;
  protected boolean extinct;
  protected int numOfWings;
  protected ArrayList<Food> dietPreference;
  protected ArrayList<String> characteristics;

  @Override
  public String getBirdName() {
    return name;
  }

  @Override
  public boolean getExtinct() {
    return extinct;
  }

  @Override
  public int getNumOfWings() {
    return numOfWings;
  }

  @Override
  public ArrayList<Food> getDietPreference() {
    return dietPreference;
  }

  @Override
  public ArrayList<String> getCharacteristics() {
    return characteristics;
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
                    + "They have the following characteristics:%s. \n",
            name, birdType, birdType, (extinct) ? "" : "not", numOfWings, dietString.toString(),
            characteristicsString.toString());
  }
}
