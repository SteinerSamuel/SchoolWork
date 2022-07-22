package questions;

import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * A test class for the question implementation.
 */
public class QuestionTest {
  private Question[] testQuestionaire;

  @Before
  public void setUp() throws Exception {
    Question t1 = new Likert("PDP is a fun class.");
    Question t2 = new MultipleChoice("What class is this?", "1", "PDP", "AI",
            "Software Engineering");
    Question t3 = new MultipleChoice("Where is this class?", "2", "Georgia",
            "Boston", "The Moon", "Earth");
    Question t4 = new MultipleSelect("This is a test question.", "1 2", "Yes",
            "Yes", "No");
    Question t5 = new TrueFalse("Is this a class?", "True");
    testQuestionaire = new Question[]{t1, t2, t3, t4, t5};
  }

  @Test
  public void testLikert() {
    // Test if Likert Questions can be made and both necessary functions work.
    Question t1 = new Likert("Are you having fun?");
    // Test if we get incorrect if we give a bad answer
    assertEquals("Incorrect", t1.answer("213"));
    assertEquals("Incorrect", t1.answer("TRUE"));
    //Test if we get correct when given a valid answer
    assertEquals("Correct", t1.answer("1"));
  }

  @Test
  public void testMultipleChoice() {
    //Test if Multiple choice questions can be made
    Question t1 = new MultipleChoice("What?", "1", "What?", "No?", "Yes?");
    try {
      Question t2 = new MultipleChoice("What?", "0", "1", "2", "#");
      fail("Should throw error");
    } catch (IllegalArgumentException e) {
      // Do nothing intended behavior.
    }
    //Test if we give an incorrect answer or invalid answer
    assertEquals("Incorrect", t1.answer("2"));
    assertEquals("Incorrect", t1.answer("1 2 3"));
    assertEquals("Incorrect", t1.answer("answer"));
    // Test if we get correct when given the valid answer
    assertEquals("Correct", t1.answer("1"));
  }

  @Test
  public void testMultipleSelection() {
    // Test if a multiple select question can be made
    Question t1 = new MultipleSelect("Who?", "1 2 3", "Doctor", "Whorton", "Mister", "sir");
    try {
      Question t2 = new MultipleSelect("Who?", "9 1", "1", "2", "3");
      fail();
    } catch (IllegalArgumentException e) {
      // Do nothing correct behavior.
    }
    // test if we get incorrect when given invalid answers
    assertEquals("Incorrect", t1.answer("1"));
    assertEquals("Incorrect", t1.answer("4"));
    assertEquals("Incorrect", t1.answer("1 2"));
    assertEquals("Incorrect", t1.answer("23"));
    assertEquals("Incorrect", t1.answer("Doctor"));
    // test if we give correct answer
    assertEquals("Correct", t1.answer("1 2 3"));
    assertEquals("Correct", t1.answer("2 3 1"));
  }

  @Test
  public void testTrueFalse() {
    //test we construct questions correctly.
    Question t1 = new TrueFalse("true?", "True");
    try {
      Question t2 = new TrueFalse("False?", "What");
      fail("Should of thrown error.");
    } catch (IllegalArgumentException e) {
      // do nothing correct behavior
    }
    // We should get incorrect for bad or invalid answers
    assertEquals("Incorrect", t1.answer("False"));
    assertEquals("Incorrect", t1.answer("Whom"));
    // WE should get correct when given valid answers
    assertEquals("Correct", t1.answer("True"));
  }

  @Test
  public void testSort() {
    StringBuilder testString = new StringBuilder();
    for (Question q : testQuestionaire) {
      testString.append(q.getText()).append('\n');
    }
    assertEquals("PDP is a fun class.\nWhat class is this?\nWhere is this class?\n"
            + "This is a test question.\nIs this a class?\n", testString.toString());

    Arrays.sort(testQuestionaire);

    testString = new StringBuilder();

    for (Question q : testQuestionaire) {
      testString.append(q.getText()).append('\n');
    }
    assertEquals("Is this a class?\nWhat class is this?\nWhere is this class?\n"
            + "This is a test question.\nPDP is a fun class.\n", testString.toString());
  }
}