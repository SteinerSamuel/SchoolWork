package questions;

import java.util.Arrays;
import java.util.List;

/**
 * A multiple selection class used to store questions which have multiple choices and multiple
 * answers.
 */
public final class MultipleSelect extends QuestionAbstract {

  /**
   * Constructor for a multiple selection question, as of right now the question does nothing with
   * the answer choices given to it this can be expanded.
   *
   * @param question The question text.
   * @param answer   The correct answer(s)
   * @param a1       choice 1 text
   * @param a2       choice 2 text
   * @param a3       choice 3 text
   */
  public MultipleSelect(String question, String answer, String a1, String a2, String a3) {
    for (String s : answer.split(" ")) {
      if (Integer.parseInt(s) < 1 || Integer.parseInt(s) > 8) {
        throw new IllegalArgumentException("One of the answers you gave are outside the range of "
                + "acceptable answers!");
      }
    }
    this.question = question;
    this.answer = answer;
  }

  public MultipleSelect(String question, String answer, String a1, String a2, String a3,
                        String a4) {
    this(question, answer, a1, a2, a3);
  }

  public MultipleSelect(String question, String answer, String a1, String a2, String a3, String a4,
                        String a5) {
    this(question, answer, a1, a2, a3);
  }

  public MultipleSelect(String question, String answer, String a1, String a2, String a3, String a4,
                        String a5, String a6) {
    this(question, answer, a1, a2, a3);
  }

  public MultipleSelect(String question, String answer, String a1, String a2, String a3, String a4,
                        String a5, String a6, String a7) {
    this(question, answer, a1, a2, a3);
  }

  public MultipleSelect(String question, String answer, String a1, String a2, String a3, String a4,
                        String a5, String a6, String a7, String a8) {
    this(question, answer, a1, a2, a3);
  }

  @Override
  public String answer(String answer) {
    boolean correct = true;
    try {
      for (String s : answer.split(" ")) {
        if (Integer.parseInt(s) < 1 || Integer.parseInt(s) > 8) {
          correct = false;
        }
      }
      List<String> l = Arrays.asList(this.answer.split(" "));
      for (String s : answer.split(" ")) {
        if (!(l.contains(s))) {
          correct = false;
          break;
        }
      }

      List<String> l2 = Arrays.asList(answer.split(" "));
      for (String s : this.answer.split(" ")) {
        if (!(l2.contains(s))) {
          correct = false;
          break;
        }
      }
    } catch (NumberFormatException e) {
      return "Incorrect";
    }

    if (correct) {
      return "Correct";
    } else {
      return "Incorrect";
    }
  }

  @Override
  public String toString() {
    return "3 " + this.question;
  }
}
