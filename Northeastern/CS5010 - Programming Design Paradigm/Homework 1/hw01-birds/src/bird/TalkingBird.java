package bird;

/**
 * An interface which is for birds which can talk.
 */
public interface TalkingBird extends Bird {
  /**
   * Gets the size of the vocab of the bird.
   *
   * @return The size of the birds vocab.
   */
  int getVocabSize();

  /**
   * gets the birds favorite phrase.
   *
   * @return The birds favorite Phrase.
   */
  String getFavoritePhrase();
}
