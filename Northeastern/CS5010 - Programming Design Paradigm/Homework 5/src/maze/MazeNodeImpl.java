package maze;

import java.util.ArrayList;

/**
 * Maze node implementation.
 */
public class MazeNodeImpl implements MazeNode {
  private final ArrayList<CardinalDirections> walls = new ArrayList<>();
  private Contents content;

  /**
   * Constructor for a Maze Node.
   *
   * @param content the contents of th nodes
   */
  public MazeNodeImpl(Contents content) {
    this.content = content;
  }

  @Override
  public void setWall(CardinalDirections d) {
    walls.add(d);
  }

  @Override
  public ArrayList<CardinalDirections> getWalls() {
    return walls;
  }

  @Override
  public Contents getContent() {
    return content;
  }

  @Override
  public void setContent(Contents content) {
    this.content = content;
  }

  @Override
  public String toString() {
    if (walls.isEmpty()) {
      return "╋"; // no walls
    } else if (walls.size() == 1) {
      switch (walls.get(0)) {
        case NORTH:
          return "┳";
        case SOUTH:
          return "┻";
        case WEST:
          return "┣";
        case EAST:
          return "┫";
        default:
          break;
      }
    } else if (walls.size() == 2) {
      if (walls.contains(CardinalDirections.EAST) && walls.contains(CardinalDirections.WEST)) {
        return "┃";
      } else if (walls.contains(CardinalDirections.EAST)
              && walls.contains(CardinalDirections.NORTH)) {
        return "┓";
      } else if (walls.contains(CardinalDirections.EAST)
              && walls.contains(CardinalDirections.SOUTH)) {
        return "┛";
      } else if (walls.contains(CardinalDirections.NORTH)
              && walls.contains(CardinalDirections.WEST)) {
        return "┏";
      } else if (walls.contains(CardinalDirections.SOUTH)
              && walls.contains(CardinalDirections.WEST)) {
        return "┗";
      } else {
        return "━";
      }
    } else if (walls.size() == 3) {
      if (!walls.contains(CardinalDirections.EAST)) {
        return "╺";
      } else if (!walls.contains(CardinalDirections.WEST)) {
        return "╸";
      } else if (!walls.contains(CardinalDirections.NORTH)) {
        return "╹";
      } else {
        return "╻";
      }
    }
    return " ";
  }

  @Override
  public int compareTo(MazeNode o) {
    return this.hashCode() - o.hashCode();
  }
}
