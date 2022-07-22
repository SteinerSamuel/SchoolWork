package bignumber;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test for the big number class.
 */
public class BigNumberImplTest {
  private BigNumber testNumber;
  private BigNumber testNumber2;
  private BigNumber testNumber3;

  @Before
  public void setUp() {
    testNumber = new BigNumberImpl();
    testNumber2 = new BigNumberImpl("200");
    testNumber3 = new BigNumberImpl("3669");
  }

  @Test
  public void testConstructors() {
    testNumber = new BigNumberImpl("123");

    try {
      testNumber = new BigNumberImpl("a");
      fail("Should throw an error!");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior.
    }
  }


  @Test
  public void testLength() {
    assertEquals(0, testNumber.length());
    assertEquals(3, testNumber2.length());
  }

  @Test
  public void testShiftLeft() {
    assertEquals("200", testNumber2.toString());
    testNumber2 = testNumber2.shiftLeft(3);
    assertEquals("200000", testNumber2.toString());

    testNumber2 = testNumber2.shiftLeft(-5);
    assertEquals("2", testNumber2.toString());
  }

  @Test
  public void testShiftRight() {
    assertEquals("200", testNumber2.toString());
    testNumber2 = testNumber2.shiftRight(2);
    assertEquals("2", testNumber2.toString());

    testNumber2 = testNumber2.shiftRight(-5);
    assertEquals("200000", testNumber2.toString());
  }

  @Test
  public void testAddDigit() {
    assertEquals("3669", testNumber3.toString());
    assertEquals("200", testNumber2.toString());
    //Test to make sure the validation works
    try {
      testNumber3.addDigit(10);
      fail("should throw error! Value passed is bigger than 1 digit.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected result
    }

    try {
      testNumber3.addDigit(-2);
      fail("Should throw error! Value passed is negative!");
    } catch (IllegalArgumentException e) {
      // Do nothing
    }

    // Test normal functionality
    testNumber2 = testNumber2.addDigit(4);
    assertEquals("204", testNumber2.toString());
    testNumber2 = testNumber2.addDigit(3);
    assertEquals("207", testNumber2.toString());

    // Testing carry over
    testNumber3 = testNumber3.addDigit(2);
    assertEquals("3671", testNumber3.toString());
  }

  @Test
  public void testGetDigitAt() {
    assertEquals(0, testNumber2.getDigitAt(0));
    assertEquals(6, testNumber3.getDigitAt(2));

    try {
      testNumber3.getDigitAt(213);
      fail("This should throw an error!");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
  }

  @Test
  public void testCopy() {
    BigNumber testNumber4 = testNumber2;
    BigNumber testNumber5 = testNumber2.copy();

    assertEquals("200", testNumber4.toString());
    assertEquals("200", testNumber5.toString());

    testNumber2.addDigit(6);

    assertEquals("206", testNumber2.toString());
    assertEquals("206", testNumber4.toString());
    assertEquals("200", testNumber5.toString());
  }


  @Test
  public void addTest() {
    assertEquals("200", testNumber2.toString());
    assertEquals("3669", testNumber3.toString());

    BigNumber tes = testNumber2.add(testNumber3);
    assertEquals("3869", tes.toString());

    //Test with empty number

    tes = testNumber2.add(testNumber);
    assertEquals("200", tes.toString());


    //Test with number which is large
    testNumber3 = new BigNumberImpl("2031341234");

    tes = testNumber2.add(testNumber3);
    assertEquals("2031341434", tes.toString());
  }

  @Test
  public void testCompre() {
    assertEquals(1, testNumber3.compareTo(testNumber2));
  }

  @Test
  public void shiftZero() {
    testNumber3 = new BigNumberImpl("00000");
    testNumber3 = testNumber3.shiftRight(1);

    assertEquals("0", testNumber3.toString());
  }
}