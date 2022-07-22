package questions;

/**
 * Multiple choice class which is used to store multiple choice questions.
 */
public final class MultipleChoice extends QuestionAbstract {

  /**
   * Constructor for a multiple choice question, as of right now the question does nothing with the
   * answer choices given to it this can be expanded.
   *
   * @param question The question.
   * @param answer   The correct answer.
   * @param a1       answer 1 choice
   * @param a2       answer 2 choice
   * @param a3       answer 3 choice.
   */
  public MultipleChoice(String question, String answer,
                        String a1, String a2, String a3) {
    if (0 < Integer.parseInt(answer) && Integer.parseInt(answer) < 9) {
      this.question = question;
      this.answer = answer;
    } else {
      throw new IllegalArgumentException("The answer should be between 1 and 8.");
    }
  }

  public MultipleChoice(String question, String answer,
                        String a1, String a2, String a3, String a4) {
    this(question, answer, a1, a2, a3);
  }

  public MultipleChoice(String question, String answer,
                        String a1, String a2, String a3, String a4, String a5) {
    this(question, answer, a1, a2, a3);
  }

  public MultipleChoice(String question, String answer,
                        String a1, String a2, String a3, String a4, String a5, String a6) {
    this(question, answer, a1, a2, a3);
  }

  public MultipleChoice(String question, String answer,
                        String a1, String a2, String a3, String a4, String a5, String a6, String a7
  ) {
    this(question, answer, a1, a2, a3);
  }

  public MultipleChoice(String question, String answer,
                        String a1, String a2, String a3, String a4, String a5, String a6, String a7,
                        String a8) {
    this(question, answer, a1, a2, a3);
  }

  @Override
  public String answer(String answer) {
    try {
      if (Integer.parseInt(answer) == Integer.parseInt(this.answer)) {
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
    return "2 " + question;
  }
}
