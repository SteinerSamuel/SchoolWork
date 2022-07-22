package weather;

import static java.lang.Math.pow;

/**  A weather reading represents a single reading of a weather station in a Stevenson Screen. */
public class WeatherReading {
  // These are given to the constructor
  private int airTemperature;
  private int dewPoint;
  private int windSpeed;
  private int totalRain;

  // these are derived from the units given tot he constructor
  private int relativeHumidity;
  private int heatIndex;
  private int windChill;

  // The following are the 9 coefficients for Heat Index (HI)
  final double heatIndexC1 = -8.78469475556;
  final double heatIndexC2 = 1.61139411;
  final double heatIndexC3 = 2.33854883889;
  final double heatIndexC4 = -0.14611605;
  final double heatIndexC5 = -0.012308094;
  final double heatIndexC6 = -0.0164248277778;
  final double heatIndexC7 = 0.002211732;
  final double heatIndexC8 = 0.00072546;
  final double heatIndexC9 = -0.000003582;

  // The following are the coefficients for Wind Chill WC
  final double windChillC1 = 35.74;
  final double windChillC2 = 0.6215;
  final double windChillC3 = -35.75;
  final double windChillC4 = 0.4275;

  /** Default constructor. The constructor takes four parameters:
   *
   *  @param airTemperature the air temperature in Celsius
   *  @param dewPoint the dew point temperature in Celsius
   *                  cannot be greater than the air temperature
   *  @param windSpeed the non-negative wind speed in miles per hour
   *  @param totalRain and the non-negative total rain received in the last 24 hours in millimeters
   *  */
  public WeatherReading(int airTemperature, int dewPoint, int windSpeed, int totalRain) {
    if (dewPoint > airTemperature) {
      throw new IllegalArgumentException("Dew point must be less than air temperature!");
    }
    if (windSpeed < 0) {
      throw new IllegalArgumentException("Wind speed must be non-negative!");
    }
    if (totalRain < 0) {
      throw new IllegalArgumentException(
              "Total amount of rain received in the last 24 hours must be non-negative!");
    }
    this.airTemperature = airTemperature;
    this.dewPoint = dewPoint;
    this.windSpeed = windSpeed;
    this.totalRain = totalRain;
    calcRelativeHumidity();
    calcHeatIndex();
    calcWindChill();
  }

  /** Helper function to convert temperature from celsius to fahrenheit.
   *
   * @param temperature The temperature you want to convert in celsius
   *
   * @return  The temperature converted to fahrenheit*/
  private int convertTemperature(int temperature) {
    return (int) (temperature * 1.8 + 32);
  }

  /**
   * Calculates and sets the relative humidity based on this function r = 5(d - t + 20) which is
   * derived from the the formula for dew point d = t - (100 - r) / 5  in these formulas d = dew
   * point t = air temperature and r is relative humidity.
   */
  private void calcRelativeHumidity() {

    this.relativeHumidity = 5 * (this.dewPoint - this.airTemperature + 20);
  }

  /**
   * Calculates and sets the heat index based on the following formula:
   * HI = c1 + c2(T) + c3(R) + c4( T * R) + c5(T^2) + c6(R^2) + c7(T^2 * R) + c8(T * R^2)
   *      + c9(T^2*R^2).
   */
  private void calcHeatIndex() {

    this.heatIndex = (int) (this.heatIndexC1 + this.heatIndexC2 * this.airTemperature
            + this.heatIndexC3 * this.relativeHumidity
            + this.heatIndexC4 * (this.airTemperature * this.relativeHumidity)
            + this.heatIndexC5 * (pow(this.airTemperature, 2))
            + this.heatIndexC6 * (pow(this.relativeHumidity, 2))
            + this.heatIndexC7 * (pow(this.airTemperature, 2) * this.relativeHumidity)
            + this.heatIndexC8 * (this.airTemperature * pow(this.relativeHumidity, 2))
            + this.heatIndexC9 * (pow(this.airTemperature, 2) * pow(this.relativeHumidity, 2)));
  }

  /** Calculates the wind chill using the following formula:
   *  WindChill = c1 + c2T + c3v^(.16) + c4Tv^(.16).
   *  where:
   *  T = Temperature in Fahrenheit (a helper function is used to convert the temperature)
   *  v = Wind Speed in mph(Miles per Hour).
   *  */
  private void calcWindChill() {
    this.windChill = (int) (this.windChillC1
            + this.windChillC2 * convertTemperature(this.airTemperature)
            + this.windChillC3 * pow(this.windSpeed, 0.16)
            + this.windChillC4 * (convertTemperature(this.airTemperature)
            * pow(this.windSpeed, 0.16)));
  }

  public int getTemperature() {
    return this.airTemperature;
  }

  public int getDewPoint() {
    return this.dewPoint;
  }

  public int getWindSpeed() {
    return this.windSpeed;
  }

  public int getTotalRain() {
    return this.totalRain;
  }

  public int getRelativeHumidity() {
    return this.relativeHumidity;
  }

  public int getHeatIndex() {
    return this.heatIndex;
  }

  public int getWindChill() {
    return this.windChill;
  }

  @Override
  public String toString() {
    return String.format("Reading: T = %d, D = %d, v = %d, rain = %d",
            this.airTemperature, this.dewPoint, this.windSpeed, this.totalRain);
  }
}
