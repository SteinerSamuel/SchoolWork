package questions;

/**
 * Concrete class which is used for like rating questions.
 */
public final class Likert extends QuestionAbstract {

  /**
   * Default constructor for Likert question.
   *
   * @param question The text of the question.
   */
  public Likert(String question) {
    super(question, "1 2 3 4 5");
  }

  @Override
  public String answer(String answer) {
    try {
      if (0 < Integer.parseInt(answer) && Integer.parseInt(answer) < 6) {
        return "Correct";
      } else {
        return "Incorrect";
      }
    } catch (NumberFormatException e) {
      return "Incorrect";
    }
  }

  @Override
  public String toString() {
    return "4 " + this.question;
  }
}