package bird;

import java.util.StringJoiner;

/**
 * An abstract class for Talking Birds which extends Bird.
 */
public abstract class TalkingBirdAbstract extends BirdAbstract implements TalkingBird {
  protected int vocabSize;
  protected String favoritePhrase;

  @Override
  public int getVocabSize() {
    return vocabSize;
  }

  @Override
  public String getFavoritePhrase() {
    return favoritePhrase;
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
                    + "They have the following characteristics:%s. %s can speak %s knows %d words, "
                    + "and %s's favorite phrase is \"%s\". +\n",
            name, birdType, birdType, (extinct) ? "" : "not", numOfWings, dietString.toString(),
            characteristicsString.toString(), name, name, vocabSize, name, favoritePhrase);
  }
}
