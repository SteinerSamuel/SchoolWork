package listadt;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * A test of the Utilities Class.
 */
public class ListADTUtilitiesTest {
  public Integer[] intArray;
  public String[] stringArray;
  public String[] stringArray2;
  public Double[] doubleArray;
  public Integer[] emptyArray;

  @Before
  public void setUp() {
    intArray = new Integer[]{1, 2, 3, 4, 5, 6, 1, 2, 3, 4};
    stringArray = new String[]{"I", "want", "to", null};
    stringArray2 = new String[]{"Hello", null, "I'm", null};
    doubleArray = new Double[]{123.123};
    emptyArray = new Integer[]{};

  }

  @Test
  public void testToList() {
    // Test if it works correctly with array of 0, 1 and many elements.
    assertEquals("()", ListADTUtilities.toList(emptyArray).toString());
    assertEquals("(123.123)", ListADTUtilities.toList(doubleArray).toString());
    assertEquals("(1 2 3 4 5 6 1 2 3 4)", ListADTUtilities.toList(intArray).toString());
  }

  @Test
  public void testToListNullValues() {
    // Test with 1 null value
    try {
      ListADTUtilities.toList(stringArray);
      fail("Should throw an IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing this is the intended behavior.
    }
    // Test with more than 1 null value
    try {
      ListADTUtilities.toList(stringArray2);
      fail("Should throw an IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing this is intended behavior
    }
  }

  @Test
  public void testAddAll() {
    // Test adding values to a list of different lengths
    ListADT<Integer> intList = ListADTUtilities.toList(intArray);
    ListADT<Double> doubleList = ListADTUtilities.toList(doubleArray);

    // add 1 and several values to a list
    ListADTUtilities.addAll(intList, 99);
    ListADTUtilities.addAll(doubleList, 22.21, 123.4, 555.55);

    assertEquals("(1 2 3 4 5 6 1 2 3 4 99)", intList.toString());
    assertEquals("(123.123 22.21 123.4 555.55)", doubleList.toString());
  }

  @Test
  public void testAddAllNull() {
    // Test to ensure exceptions are thrown
    ListADT<Integer> intList = ListADTUtilities.toList(intArray);

    try {
      ListADTUtilities.addAll(intList, 1, null, 123);
      fail("IllegalArgumentException should be thrown.");
    } catch (IllegalArgumentException e) {
      // Assure no values were added to list.
      assertEquals("(1 2 3 4 5 6 1 2 3 4)", intList.toString());
    }

    try {
      ListADTUtilities.addAll(intList, null, null, 1);
      fail("IllegalArgumentException should be thrown.");
    } catch (IllegalArgumentException e) {
      // Assure no values were added to list.
      assertEquals("(1 2 3 4 5 6 1 2 3 4)", intList.toString());
    }
  }

  @Test
  public void testFrequency() {
    ListADT<Double> doubleList = ListADTUtilities.toList(doubleArray);
    ListADT<Integer> integerList = ListADTUtilities.toList(intArray);

    // Test frequency several cases
    assertEquals(0, ListADTUtilities.frequency(doubleList, 88.88));
    assertEquals(2, ListADTUtilities.frequency(integerList, 4));
    assertEquals(1, ListADTUtilities.frequency(integerList, 5));
  }


  @Test
  public void testDisjoint() {
    ListADT<Integer> intDisjoint1 = ListADTUtilities.toList(new Integer[]{3, 1, 3, 1, 1, 6, 4});
    ListADT<Integer> intDisjoint2 = ListADTUtilities.toList(new Integer[]{99, 98, 97, 96});

    // The two lists above should be considered disjointed
    assertTrue(ListADTUtilities.disjoint(intDisjoint1, intDisjoint2));

    // The list intDisjoint1 and intList share the values 1, 3, 4 they should not be considered
    // disjointed
    ListADT<Integer> intList = ListADTUtilities.toList(intArray);
    assertFalse(ListADTUtilities.disjoint(intDisjoint1, intList));
  }

  @Test
  public void testDisjointNullValue() {
    ListADT<Integer> nullList = new ListADTImpl<Integer>();
    nullList.addBack(null);
    nullList.addBack(2);
    ListADT<Integer> nullList2 = new ListADTImpl<Integer>();
    nullList2.addBack(3);
    nullList2.addBack(null);
    nullList2.addBack(null);

    //Test if it throws an exception when there is a null value in first parameter
    ListADT<Integer> intList = ListADTUtilities.toList(intArray);
    try {
      ListADTUtilities.disjoint(nullList, intList);
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    //Test to see if it throws an exception when in parameter 2
    try {
      ListADTUtilities.disjoint(intList, nullList2);
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    //Test to assure if both lists contain null an exception is thrown 
    try {
      ListADTUtilities.disjoint(nullList, nullList2);
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }
  }

  @Test
  public void testEquals() {
    ListADT<Integer> intList1 = ListADTUtilities.toList(intArray);
    ListADT<Integer> intList2 = ListADTUtilities.toList(intArray);
    ListADT<Integer> intList3 = ListADTUtilities.toList(new Integer[]{1, 2, 3});
    ListADT<Integer> intList4 = ListADTUtilities.toList(
            new Integer[]{1, 2, 3, 4, 5, 6, 9, 2, 3, 4});

    // check to see it returns true when bot lists are equal
    assertTrue(ListADTUtilities.equals(intList1, intList2));

    // check to see if it returns false when lists are of different size
    assertFalse(ListADTUtilities.equals(intList3, intList1));

    // check to see if it returns false when lists are of same size but not equal
    assertFalse(ListADTUtilities.equals(intList1, intList4));
  }

  @Test
  public void testEqualsNullValue() {
    ListADT<Integer> nullList = new ListADTImpl<Integer>();
    nullList.addBack(null);
    nullList.addBack(2);
    ListADT<Integer> nullList2 = new ListADTImpl<Integer>();
    nullList2.addBack(3);
    nullList2.addBack(null);
    nullList2.addBack(null);

    //Test if it throws an exception when there is a null value in first parameter
    ListADT<Integer> intList = ListADTUtilities.toList(intArray);
    try {
      ListADTUtilities.equals(nullList, intList);
      fail("Should throw IllegalArgumentException.");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    //Test if it throws an exception when there is a null value in the second parameter
    try {
      ListADTUtilities.equals(intList, nullList2);
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

    //Test to see if if exception is thrown when bot lists contain null values.
    try {
      ListADTUtilities.equals(nullList, nullList2);
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      // Do nothing expected behavior
    }

  }

  @Test
  public void testSwap() {
    ListADT<Integer> intList = ListADTUtilities.toList(intArray);

    ListADTUtilities.swap(intList, 0, 3);
    assertEquals("(4 2 3 1 5 6 1 2 3 4)", intList.toString());
  }

  @Test
  public void testSwapOutOfBounds() {
    ListADT<Integer> intList = ListADTUtilities.toList(intArray);

    // Check if giving an out of bound number for i throws an error
    try {
      ListADTUtilities.swap(intList, 30, 2);
      fail("IndexOutOfBoundException expected i is out of bounds.");
    } catch (IndexOutOfBoundsException e) {
      // Do nothing expected behavior
    }

    //Check if giving an out of bound number for j throws an error
    try {
      ListADTUtilities.swap(intList, 2, 30);
      fail("IndexOutOfBoundException expected j is out of bounds.");
    } catch (IndexOutOfBoundsException e) {
      // Do nothing expected behavior
    }

  }

  @Test
  public void testReverseEvenNumberElements() {
    ListADT<Integer> intList = ListADTUtilities.toList(new Integer[]{1, 3, 5, 7});

    ListADTUtilities.reverse(intList);

    assertEquals("(7 5 3 1)", intList.toString());
  }

  @Test
  public void testReverseOddNumberElements() {
    ListADT<Integer> intList = ListADTUtilities.toList(new Integer[]{0, 2, 4, 6, 8});

    ListADTUtilities.reverse(intList);

    assertEquals("(8 6 4 2 0)", intList.toString());
  }
}