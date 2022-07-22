package lookandsay;

import java.math.BigInteger;

/**
 * Look and Say Iterator, is an implementation of the look and say sequence which is a coded way of
 * representing how we would say a sequence of numbers. For example 123 would be read as
 * one 1 one 2 one 3 this can be coded as 111213, which would be the next number in the sequence.
 * This implementation allows for forward or reverse travel in a sequence.
 */
public class LookAndSayIterator implements RIterator<BigInteger> {
  private static final BigInteger DEFAULT_SEED = new BigInteger("1");
  private static final BigInteger DEFAULT_END = new BigInteger("9".repeat(100));
  private final BigInteger end;
  private BigInteger curr;
  private BigInteger prev;

  /**
   * Constructor for LookAndSayIterator which takes a seed and an end, a seed must not contain any 0
   * and must not be negative and must be greater than end.
   *
   * @param seed The seed to start the sequence at
   * @param end  The end of the sequence
   */
  public LookAndSayIterator(BigInteger seed, BigInteger end) {
    // Validation
    if (seed.compareTo(BigInteger.valueOf(0)) < 0) {
      throw new IllegalArgumentException("The seed must be a positive number.");
    } else if (seed.compareTo(end) > -1) {
      throw new IllegalArgumentException("The seed must be less than the end.");
    } else if (seed.toString().contains("0")) {
      throw new IllegalArgumentException("The seed must not contain zeros");
    }
    this.end = end;
    this.curr = seed;
    this.prev = seed;
  }

  /**
   * Constructor where a seed is provided but uses the default end of 999999999.
   *
   * @param seed the seed to start the sequence at
   */
  public LookAndSayIterator(BigInteger seed) {
    this(seed, DEFAULT_END);
  }

  /**
   * A constructor which no vlaues for seed or end are given uses the default values of 1 and
   * 999999999.
   */
  public LookAndSayIterator() {
    this(DEFAULT_SEED, DEFAULT_END);
  }

  @Override
  public boolean hasPrevious() {
    // The current 'length' of the number must be a even number in order for a previous value to
    // exist;
    return prev.toString().length() % 2 == 0;
  }

  @Override
  public BigInteger prev() {
    StringBuilder prevString = new StringBuilder();
    String currString = prev.toString();

    while (currString.length() > 0) {
      int times = Integer.parseInt(currString.substring(0, 1));
      String value = currString.substring(1, 2);

      prevString.append(value.repeat(times));
      currString = currString.substring(2);
    }
    curr = prev;
    prev = new BigInteger(prevString.toString());
    return prev;
  }

  @Override
  public boolean hasNext() {
    // generate next > end? (w/o mutating)
    return curr.compareTo(end) < 1;
  }

  @Override
  public BigInteger next() {
    prev = curr;
    curr = nextHelper();
    return prev;
  }

  /**
   * Helper function to keep code clean. Calculates the next value of the current node without
   * mutation.
   *
   * @return the next value
   */
  private BigInteger nextHelper() {
    String currString = curr.toString();
    StringBuilder nextString = new StringBuilder();

    char val = currString.charAt(0);
    currString = currString.substring(1) + '.'; // Adding this . as an ending character saves 1 line
    int count = 1;

    for (char testVal : currString.toCharArray()) {
      if (testVal == val) {
        count++;
      } else {
        nextString.append(count).append(val);
        count = 1;
        val = testVal;
      }
    }
    return new BigInteger(nextString.toString());
  }
}
