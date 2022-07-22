package questions;

/**
 * Question is an interface which represents a question. This questions should be able to allow
 * users to submit an answer and get feedback on whether or not the answer is Correct or Incorrect.
 */
public interface Question extends Comparable<Question> {
  /**
   * Gets the quesiton text and returns it to the user.
   *
   * @return The question text.
   */
  String getText();

  /**
   * Given the users answer returns whether or not the user was correct.
   *
   * @param answer The answer of the user.
   * @return "Correct" or "Incorrect" based on users answer.
   */
  String answer(String answer);
}
