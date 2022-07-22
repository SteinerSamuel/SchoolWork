package questions;

/**
 * Abstract class for the Question implementation.
 */
public abstract class QuestionAbstract implements Question {
  protected String question;
  protected String answer;

  /**
   * Base constructor.
   *
   * @param question The question text.
   * @param answer   The answer of the question.
   */
  protected QuestionAbstract(String question, String answer) {
    this.question = question;
    this.answer = answer;
  }

  /**
   * Default empty constructor.
   */
  protected QuestionAbstract() {
    // Default empty constructor needed for concrete classes.
  }

  @Override
  public String getText() {
    return question;
  }

  @Override
  public int compareTo(Question o) {
    return this.toString().compareTo(o.toString());
  }

  /**
   * The question and a question code used for sorting.
   *
   * @return The question code based on the question type and the text of the question.
   */
  @Override
  public String toString() {
    return "0 " + question;
  }
}
