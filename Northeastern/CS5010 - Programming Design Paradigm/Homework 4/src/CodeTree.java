import java.util.HashMap;

/**
 * A  code tree which is used to create a hoffman map.
 */
public interface CodeTree {
  /**
   * Increases the frequency.
   */
  void increaseFreq();

  /**
   * Gets the frequency.
   *
   * @return the frequency that the token appears
   */
  int getFreq();

  /**
   * Returns the encoded representation.
   *
   * @return the coded representation
   */
  String getRepresentation();

  /**
   * Sets the encoded representation.
   *
   * @param rep the coded representation
   */
  void setRepresentation(String rep);

  /**
   * Gets the token.
   *
   * @return The token
   */
  String getToken();

  /**
   * Generates and returns a HashMap dictionary with token -> representation as key value pairs.
   *
   * @return Token -> Representation  dictionary.
   */
  HashMap<String, String> returnMapDict();
}