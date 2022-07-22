package questions;

/**
 * True false question which is used to store true false questions.
 */
public final class TrueFalse extends QuestionAbstract {

  /**
   * A true false question.
   *
   * @param question The question text.
   * @param answer   The correct answer.
   */
  public TrueFalse(String question, String answer) {
    if (answer.equals("True") || answer.equals("False")) {
      this.question = question;
      this.answer = answer;
    } else {
      throw new IllegalArgumentException("The answer must be eiter \"True\" or \"False\".");
    }
  }

  @Override
  public String answer(String answer) {
    if (answer.equals(this.answer)) {
      return "Correct";
    } else {
      return "Incorrect";
    }
  }

  @Override
  public String toString() {
    return "1 " + this.question;
  }
}
