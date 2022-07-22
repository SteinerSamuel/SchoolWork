package bst;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;


/**
 * Test for binary search tree.
 */
public class BSTTest {
  private BSTImpl<Integer> testBST;

  @Before
  public void setUp() {
    testBST = new BSTImpl<Integer>();
  }

  @Test
  public void testAdd() {
    assertEquals("[]", testBST.toString());
    testBST.add(10);
    assertEquals("[10]", testBST.toString());
    testBST.add(3);
    testBST.add(5);
    assertEquals("[3 5 10]", testBST.toString());
  }

  @Test
  public void testAddDuplicate() {
    assertEquals(0, testBST.size());
    testBST.add(10);
    testBST.add(10);
    assertEquals(1, testBST.size());
  }

  @Test
  public void testSize() {
    assertEquals(0, testBST.size());
    testBST.add(10);
    testBST.add(2);
    assertEquals(2, testBST.size());
  }

  @Test
  public void testPresent() {
    testBST.add(1);
    testBST.add(2);
    testBST.add(3);
    testBST.add(4);

    assertTrue(testBST.present(1));
    assertFalse(testBST.present(0));
  }

  @Test
  public void testMinimum() {
    testBST.add(1);
    testBST.add(12);
    testBST.add(3);
    testBST.add(41);
    testBST.add(11);

    assertEquals((Integer) 1, testBST.minimum());
  }

  @Test
  public void testMinimumEmpty() {
    assertNull(testBST.minimum());
  }

  @Test
  public void testMaximum() {
    testBST.add(1);
    testBST.add(12);
    testBST.add(3);
    testBST.add(41);
    testBST.add(11);

    assertEquals((Integer) 41, testBST.maximum());
  }

  @Test
  public void testMaximumEmpty() {
    assertNull(testBST.maximum());
  }


  @Test
  public void testPreOrder() {
    testBST.add(1);
    testBST.add(12);
    testBST.add(3);
    testBST.add(41);
    testBST.add(11);

    assertEquals("[1 12 3 11 41]", testBST.preOrder());

  }

  @Test
  public void testInOrder() {
    testBST.add(1);
    testBST.add(12);
    testBST.add(3);
    testBST.add(41);
    testBST.add(11);

    assertEquals("[1 3 11 12 41]", testBST.inOrder());
  }

  @Test
  public void testPostOrder() {
    testBST.add(1);
    testBST.add(12);
    testBST.add(3);
    testBST.add(41);
    testBST.add(11);

    assertEquals("[11 3 41 12 1]", testBST.postOrder());
  }

  @Test
  public void height() {
    testBST.add(1);
    testBST.add(12);
    testBST.add(3);
    testBST.add(41);
    testBST.add(11);

    assertEquals(4, testBST.height());
  }
}