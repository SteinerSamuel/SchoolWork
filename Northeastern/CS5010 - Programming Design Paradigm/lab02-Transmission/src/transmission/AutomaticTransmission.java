package transmission;

/**
 * Automatic transmission clas which implements the Transmission interface. The gear is
 * automatically designated by the speed.
 */
public class AutomaticTransmission implements Transmission {
  private final int breakPoint1;
  private final int breakPoint2;
  private final int breakPoint3;
  private final int breakPoint4;
  private final int breakPoint5;

  private final int speed;
  private final int gear;

  /**
   * Default constructor this will make a new transmission at speed of 0 and gear 0. Takes 5
   * breakpoints used to decide which gear the transmission is in.
   *
   * @param breakPoint1 The breakpoint between gear 1 and 2.
   * @param breakPoint2 The breakpoint between gear 2 and 3.
   * @param breakPoint3 The breakpoint between gear 3 and 4.
   * @param breakPoint4 The breakpoint between gear 4 and 5.
   * @param breakPoint5 The breakpoint between gear 5 and 6.
   */
  public AutomaticTransmission(int breakPoint1, int breakPoint2, int breakPoint3,
                               int breakPoint4, int breakPoint5) {
    if (0 < breakPoint1 && breakPoint1 < breakPoint2 && breakPoint2 < breakPoint3
            && breakPoint3 < breakPoint4 && breakPoint4 < breakPoint5) {
      this.breakPoint1 = breakPoint1;
      this.breakPoint2 = breakPoint2;
      this.breakPoint3 = breakPoint3;
      this.breakPoint4 = breakPoint4;
      this.breakPoint5 = breakPoint5;
      this.speed = 0;
      this.gear = 0;
    } else {
      throw new IllegalArgumentException("Each subsequent should be larger than the previous. "
              + "Such that breakPoint1 < breakPoint2 < breakPoint3 < breakpoint4 < breakPoint5. ");
    }
  }

  /**
   * A constructor which is not at 0 speed. This constructor will automatically put the transmission
   * in the correct gear.
   *
   * @param breakPoint1 The breakpoint between gear 1 and 2.
   * @param breakPoint2 The breakpoint between gear 2 and 3.
   * @param breakPoint3 The breakpoint between gear 3 and 4.
   * @param breakPoint4 The breakpoint between gear 4 and 5.
   * @param breakPoint5 The breakpoint between gear 5 and 6.
   * @param speed       The speed the transmission is in.
   */
  public AutomaticTransmission(int breakPoint1, int breakPoint2, int breakPoint3,
                               int breakPoint4, int breakPoint5, int speed) {
    if (0 < breakPoint1 && breakPoint1 < breakPoint2 && breakPoint2 < breakPoint3
            && breakPoint3 < breakPoint4 && breakPoint4 < breakPoint5) {
      this.breakPoint1 = breakPoint1;
      this.breakPoint2 = breakPoint2;
      this.breakPoint3 = breakPoint3;
      this.breakPoint4 = breakPoint4;
      this.breakPoint5 = breakPoint5;
    } else {
      throw new IllegalArgumentException("Each subsequent should be larger than the previous. "
              + "Such that breakPoint1 < breakPoint2 < breakPoint3 < breakpoint4 < breakPoint5. ");
    }
    if (speed >= 0) {
      this.speed = speed;
      if (this.speed == 0) {
        this.gear = 0;
      } else if (this.speed < this.breakPoint1) {
        this.gear = 1;
      } else if (this.speed < this.breakPoint2) {
        this.gear = 2;
      } else if (this.speed < this.breakPoint3) {
        this.gear = 3;
      } else if (this.speed < this.breakPoint4) {
        this.gear = 4;
      } else if (this.speed < this.breakPoint5) {
        this.gear = 5;
      } else {
        this.gear = 6;
      }
    } else {
      throw new IllegalStateException("Speed may not be lower than 0.");
    }
  }


  @Override
  public int getSpeed() {
    return speed;
  }

  @Override
  public int getGear() {
    return gear;
  }

  @Override
  public Transmission increaseSpeed() {
    return new AutomaticTransmission(this.breakPoint1, this.breakPoint2, this.breakPoint3,
            this.breakPoint4, this.breakPoint5, this.speed + 2);
  }

  @Override
  public Transmission decreaseSpeed() {
    return new AutomaticTransmission(this.breakPoint1, this.breakPoint2, this.breakPoint3,
            this.breakPoint4, this.breakPoint5, this.speed - 2);
  }

  /**
   * Returns the state of the transmission in the format Transmission (speed = %d, gear = %d).
   */
  @Override
  public String toString() {
    return String.format("Transmission (speed = %d, gear = %d)", this.speed, this.gear);
  }
}
