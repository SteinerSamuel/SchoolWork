package weather;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Testing for the Weather Reading Test.
 */
public class WeatherReadingTest {
  private WeatherReading testReading;

  @Before
  public void setUp() {
    testReading = new WeatherReading(23, 12, 3, 12);
  }

  /**
   * The following tests should test the getter methods which are given to the object on
   * construction these just have to be tested for accuracy in returning the value given.
   **/
  @Test
  public void testGetTemperature() {
    assertEquals(23, testReading.getTemperature());
  }

  @Test
  public void testGetDewPoint() {
    assertEquals(12, testReading.getDewPoint());
  }

  @Test
  public void testGetWindSpeed() {
    assertEquals(3, testReading.getWindSpeed());
  }

  @Test
  public void testGetTotalRain() {
    assertEquals(12, testReading.getTotalRain());
  }

  @Test
  public void testToString() {
    String expectedOutput = "Reading: T = 23, D = 12, v = 3, rain = 12";

    assertEquals(expectedOutput, testReading.toString());
  }

  /**
   * The next set of  tests will check the invalidArgumentError which should be thrown if the given
   * arguments do not follow the necessary rules for the class.
   */
  @Test
  public void testInvalidArgumentDewPoint() {
    // Dew point should throw an illegal argument
    try {
      testReading = new WeatherReading(12, 10, 3, 12);
      // normal test for weather reading
    } catch (IllegalArgumentException e) {
      fail("An exception should not have been thrown.");
    }

    try {
      testReading = new WeatherReading(10, 12, 3, 12);
      fail("There should of been an error thrown dew point is higher than air temperature.");
    } catch (IllegalArgumentException e) {
      // If the error is thrown the illegal argument error is working.
    }

    //also if the values are equal
    try {
      testReading = new WeatherReading(10, 10, 3, 12);
    } catch (IllegalArgumentException e) {
      fail("Dew point is not higher than air temperature an error should not have been thrown.");
    }
  }

  // neither wind speed or total rain should be allowed to have a negative value
  @Test
  public void testInvalidArgumentNegativeValue() {
    try {
      testReading = new WeatherReading(12, 10, -1, 10);
      fail("The wind speed is negative and an error should be thrown.");
    } catch (IllegalArgumentException e) {
      // If error is thrown then code is working as intended
    }

    // Testing total rain
    try {
      testReading = new WeatherReading(12, 10, 0, -12);
      fail("The total rain is negative and should have failed.");
    } catch (IllegalArgumentException e) {
      // if error is thrown code is working as intended.
    }
  }

  /**
   * The next set of tests will tes the derived values being: Relative Humidity Heat Index Wind
   * Chill These have to be calculated. For the values T: 23 D: 12 WindSpeed: 3 Total Rain: 12 the
   * values are as follows: Relative Humidity:  45 Heat Index: 25 Wind Chill: 75.
   */
  @Test
  public void testGetRelativeHumidity() {
    assertEquals(45, testReading.getRelativeHumidity());
  }

  @Test
  public void testGetHeatIndex() {
    assertEquals(25, testReading.getHeatIndex());
  }

  @Test
  public void testGetWindChill() {
    assertEquals(75, testReading.getWindChill());
  }

}