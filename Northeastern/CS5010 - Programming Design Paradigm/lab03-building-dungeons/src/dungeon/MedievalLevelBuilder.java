package dungeon;

/**
 * A builder class which builds a level class this will take targets and ensure thata those targets
 * are honored.
 */
public class MedievalLevelBuilder {
  private final int level;
  private final String[] rooms;
  private final Object[][] monsters;
  private final Object[][] treasures;
  private int monstersAdded = 0;
  private int treasuresAdded = 0;

  /**
   * The default constructor for the MedievalLevelBuilder.
   *
   * @param level     The level this builder is building.
   * @param rooms     The target number of rooms.
   * @param monsters  The target number of monsters.
   * @param treasures The target number of treasures.
   */
  public MedievalLevelBuilder(int level, int rooms, int monsters, int treasures) {
    if (level < 0) {
      throw new IllegalArgumentException("Level cannot be negative!");
    } else if (rooms < 0) {
      throw new IllegalArgumentException("Expected number of rooms cannot be negative.");
    } else if (monsters < 0) {
      throw new IllegalArgumentException("Expected number of monster cannot be negative.");
    } else if (treasures < 0) {
      throw new IllegalArgumentException("Expected number of treasures cannot be negative.");
    }
    this.level = level;
    this.rooms = new String[rooms];
    this.monsters = new Object[monsters][2];
    this.treasures = new Object[treasures][2];
  }

  /**
   * This is a helper function which is used to calculate the number of non-null values in an
   * array.
   *
   * @param array The array you want to find the amount of non-null vlaues for.
   * @return The number of of non-null values should be 0 <= x <= array.length.
   */
  private int getNonNullLength(Object[] array) {
    int notNull = 0;
    for (Object o : array) {
      if (o != null) {
        notNull++;
      }
    }
    return notNull;
  }

  /**
   * Adds a room with a description to the level.
   *
   * @param description The description of the room.
   * @throws IllegalArgumentException Thrown when the number of rooms is more than the original
   *                                  target
   */
  public void addRoom(String description) {
    if (getNonNullLength(this.rooms) == this.rooms.length) {
      throw new IllegalStateException("You cannot add more rooms than originally specified!");
    } else {
      this.rooms[getNonNullLength(this.rooms)] = (description);
    }
  }

  /**
   * Adds a goblin to the level given a specified amount and room index starting from 0.
   *
   * @param index The room index starts from 0.
   * @param count The number of goblins.
   * @throws IllegalArgumentException  When the index given has no room which has been initiated.
   * @throws IllegalStateException     When the number of goblins added would make the number of
   *                                   monsters surpass the target amount of monsters.
   * @throws IndexOutOfBoundsException When the index given is bigger than the size of the array.
   */
  public void addGoblins(int index, int count) {
    if (this.rooms[index] == null) {
      throw new IllegalArgumentException("The room which was specified has not been created!");
    } else if (this.monstersAdded + count > this.monsters.length) {
      throw new IllegalStateException("The number of goblins you wanted to add would surpass the "
              + "target!");
    } else {
      for (int i = 1; i <= count; i++) {
        this.monsters[this.monstersAdded][0] = index;
        this.monsters[this.monstersAdded][1] = new Monster("goblin",
                "mischievous and very unpleasant, vengeful, and greedy creature whose "
                        + "primary purpose is to cause trouble to humankind", 7);
        this.monstersAdded++;
      }
    }
  }

  /**
   * Add an orc to the level in the specified room.
   *
   * @param index The index of the room starting from 0.
   * @throws IllegalArgumentException  When the index given has no room which has been initiated.
   * @throws IllegalStateException     When adding an orc would make the number of monsters surpass
   *                                   the target amount of monsters.
   * @throws IndexOutOfBoundsException When the index given is bigger than the size of the array.
   */
  public void addOrc(int index) {
    if (this.rooms[index] == null) {
      throw new IllegalArgumentException("The room which was specified has not been created!");
    } else if (this.monstersAdded + 1 > this.monsters.length) {
      throw new IllegalStateException("Adding an orc would make you surpass your target!");
    } else {
      this.monsters[this.monstersAdded][0] = index;
      this.monsters[this.monstersAdded][1] = new Monster("orc",
              "brutish, aggressive, malevolent being serving evil", 20);
      this.monstersAdded++;

    }
  }

  /**
   * Add an ogre to the level in the specified room.
   *
   * @param index The index of the room starting from 0.
   * @throws IllegalArgumentException  When the index given has no room which has been initiated.
   * @throws IllegalStateException     When adding an ogre would make the number of monsters surpass
   *                                   the target amount of monsters.
   * @throws IndexOutOfBoundsException When the index given is bigger than the size of the array.
   */
  public void addOgre(int index) {
    if (this.rooms[index] == null) {
      throw new IllegalArgumentException("The room which was specified has not been created!");
    } else if (this.monstersAdded + 1 > this.monsters.length) {
      throw new IllegalStateException("The adding an ogre to the level would surpass the target!");
    } else {
      this.monsters[this.monstersAdded][0] = index;
      this.monsters[this.monstersAdded][1] = new Monster("ogre",
              "large, hideous man-like being that likes to eat humans for lunch", 50);
      this.monstersAdded++;

    }
  }

  /**
   * Adds a human to the level to the specified room, for adding a a name description and the
   * hitpoints must be provided.
   *
   * @param index       The room which to add the human stating at 0.
   * @param name        The name of the human.
   * @param description The description for the human.
   * @param hitpoints   The hitpoints the human has.
   */
  public void addHuman(int index, String name, String description, int hitpoints) {
    if (this.rooms[index] == null) {
      throw new IllegalArgumentException("The room which was specified has not been created!");
    } else if (this.monstersAdded + 1 > this.monsters.length) {
      throw new IllegalStateException("The adding an ogre to the level would surpass the target!");
    } else {
      this.monsters[this.monstersAdded][0] = index;
      this.monsters[this.monstersAdded][1] = new Monster(name, description, hitpoints);
      this.monstersAdded++;
    }
  }

  /**
   * Adds a potion to the specified room.
   *
   * @param index The room number starting with room 0.
   */
  public void addPotion(int index) {
    if (this.rooms[index] == null) {
      throw new IllegalArgumentException("The room which was specified has not been created!");
    } else if (this.treasuresAdded + 1 > this.treasures.length) {
      throw new IllegalStateException("Adding a potion would surpass the treasure target.");
    } else {
      this.treasures[this.treasuresAdded][0] = index;
      this.treasures[this.treasuresAdded][1] = new Treasure("a healing potion", 1);
      this.treasuresAdded++;
    }
  }

  /**
   * Adds gold of specified value to specified room.
   *
   * @param index The room number starting with 0.
   * @param value The value of the gold.
   */
  public void addGold(int index, int value) {
    if (this.rooms[index] == null) {
      throw new IllegalArgumentException("The room which was specified has not been created!");
    } else if (this.treasuresAdded + 1 > this.treasures.length) {
      throw new IllegalStateException("Adding gold would surpass the treasure target");
    } else {
      this.treasures[this.treasuresAdded][0] = index;
      this.treasures[this.treasuresAdded][1] = new Treasure("pieces of gold", value);
      this.treasuresAdded++;
    }
  }

  /**
   * Adds the specified weapon to the specified room.
   *
   * @param index  The room to add the weapon to starting with 0.
   * @param weapon The weapon to add to the room.
   */
  public void addWeapon(int index, String weapon) {
    if (this.rooms[index] == null) {
      throw new IllegalArgumentException("The room which was specified has not been created!");
    } else if (this.treasuresAdded + 1 > this.treasures.length) {
      throw new IllegalStateException("Adding the weapon would surpass the treasure target.");
    } else {
      this.treasures[this.treasuresAdded][0] = index;
      this.treasures[this.treasuresAdded][1] = new Treasure(weapon, 10);
      this.treasuresAdded++;
    }
  }

  /**
   * Add a special item of specified description and specified value to a specified room.
   *
   * @param index       The room to add the item to.
   * @param description The description of the item.
   * @param value       The value of the item.
   */
  public void addSpecial(int index, String description, int value) {
    if (this.rooms[index] == null) {
      throw new IllegalArgumentException("The room which was specified has not been created!");
    } else if (this.treasuresAdded + 1 > this.treasures.length) {
      throw new IllegalStateException("Adding an item will surpass the target");
    } else {
      this.treasures[this.treasuresAdded][0] = index;
      this.treasures[this.treasuresAdded][1] = new Treasure(description, value);
      this.treasuresAdded++;
    }
  }

  /**
   * Builds the level once it is complete.
   *
   * @return The finished level.
   */
  public Level build() {
    if (getNonNullLength(this.rooms) == this.rooms.length
            && this.treasuresAdded == this.treasures.length
            && this.monstersAdded == this.monsters.length) {
      Level builtLevel = new Level(this.level);

      for (String s : this.rooms) {
        builtLevel.addRoom(s);
      }

      for (Object[] monster : this.monsters) {
        builtLevel.addMonster((int) monster[0], (Monster) monster[1]);
      }

      for (Object[] treasure : this.treasures) {
        builtLevel.addTreasure((int) treasure[0], (Treasure) treasure[1]);
      }


      return builtLevel;
    } else {
      throw new IllegalStateException("The level is not complete please meet all goals for"
              + " the level.");
    }
  }
}