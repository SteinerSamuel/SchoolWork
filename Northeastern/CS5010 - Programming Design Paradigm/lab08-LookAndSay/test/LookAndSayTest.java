import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.math.BigInteger;

import lookandsay.LookAndSayIterator;
import lookandsay.RIterator;

import org.junit.Test;

/**
 * Test class for look and say.
 */
public class LookAndSayTest {
  // Validation testing
  @Test
  public void validSeedLargerThanGivenEnd() {
    try {
      RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("12"),
              new BigInteger("1"));
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
  }

  @Test
  public void validSeedZeroValueGivenEnd() {
    try {
      RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("102"),
              new BigInteger("1"));
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
  }

  @Test
  public void validSeedNegativeValueGivenEnd() {
    try {
      RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("-12"),
              new BigInteger("1"));
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
  }


  @Test
  public void validSeedLargerThanDefaultEnd() {
    try {
      RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("1".repeat(111)));
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
  }

  @Test
  public void validSeedZeroValueDefaultEnd() {
    try {
      RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("102"));
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
  }

  @Test
  public void validSeedNegativeValueDefaultEnd() {
    try {
      RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("-12"));
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
  }

  // Functionality Testing

  // Has Next
  @Test
  public void hasNextDefaultTest() {
    RIterator<BigInteger> it = new LookAndSayIterator();
    assertTrue(it.hasNext());
    it.next();
    assertTrue(it.hasNext());
  }

  @Test
  public void hasNextCurrentEqualsEndTest() {
    RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("11"), new BigInteger("21"));
    it.next(); // the next node will 21 which should be but after this it should return false
    assertTrue(it.hasNext());
    it.next(); // next node will be greater than 21
    assertFalse(it.hasNext());
  }

  @Test
  public void hasNextNonDefaultEndTest() {
    RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("732183214"));
    it.next();
    assertTrue(it.hasNext());
    it.next();
    assertTrue(it.hasNext());
  }

  //Next
  @Test
  public void nextDefaultTest() {
    RIterator<BigInteger> it = new LookAndSayIterator();
    BigInteger num = it.next();
    assertEquals(new BigInteger("1"), num);
    num = it.next();
    assertEquals(new BigInteger("11"), num);
  }

  @Test
  public void nextNonDefaultTest() {
    RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("444422"));
    BigInteger num = it.next();
    assertEquals(new BigInteger("444422"), num);
    num = it.next();
    assertEquals(new BigInteger("4422"), num);
  }

  // Has Prev
  @Test
  public void hasPrevDefaultTrueTest() {
    //After first two value has prev should be true
    RIterator<BigInteger> it = new LookAndSayIterator();
    it.next();
    it.next();
    assertTrue(it.hasPrevious());
  }

  @Test
  public void hasPrevDefaultFalseTest() {
    //First two value has prev should be false
    RIterator<BigInteger> it = new LookAndSayIterator();
    assertFalse(it.hasPrevious());
    it.next();
    assertFalse(it.hasPrevious());
  }

  @Test
  public void hasPrevCustomSeedBeforeSeedEvenTest() {
    // if the custom seed has a even length than has prev should return true
    RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("444422"));
    assertTrue(it.hasPrevious());
  }

  @Test
  public void hasPrevCustomSeedOddTest() {
    // if the custom seed has a even length than has prev should return true
    RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("44422"));
    assertFalse(it.hasPrevious());
  }

  // Prev
  @Test
  public void prevDefaultTrueTest() {
    //After first two value has prev should be true
    RIterator<BigInteger> it = new LookAndSayIterator();
    it.next();
    BigInteger num = it.next();
    assertEquals(new BigInteger("11"), num);
    num = it.prev();
    assertEquals(new BigInteger("1"), num);
  }

  @Test
  public void PrevCustomSeedBeforeSeedEvenTest() {
    // if the custom seed has a even length than has prev should return true
    RIterator<BigInteger> it = new LookAndSayIterator(new BigInteger("444422"));
    assertEquals(new BigInteger("4444444422"), it.prev());
  }
}
