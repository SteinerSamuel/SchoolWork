# HomeWork 03 Mazes

This project generates a maze and allows the user to move a character within the maze to the goal.

The problem statement had several conditions which must be met.

```
    There is one goal location cell

    20% of locations (at random) have gold coins in them that the player can pick up

    10% of locations (at random) have a thief that takes some of the player's gold coins

    The maze can produce the player's location and gold count

    The maze can produce the possible moves of the player (North, South, East or West) from their current location

    The player can make a move by specifying a direction

    The player should be able to collect gold such that
        a player "collects" gold by entering a room that has gold which is then removed from the room
        a player "loses" 10% of their gold by entering a room with a thief

Your maze should not provide a way to dump the maze to the screen for anything other debugging purposes.

In addition, develop a driver program that uses command line arguments to:

    specify the size of the maze

    specify whether the maze is perfect or a non-perfect maze, and whether it is wrapping or non-wrapping
        if it is non-perfect, specify the number of walls remaining

    specify the starting point of the player

    specify the goal location
```

These are all handled in the implementation of the Maze interface.

The maze uses co-ordinates to describe locations the north most row and west most column are the 0th
row and 0th column.

The maze goal is handled as co-ordinates and only handles one set of co-ordinates this means that 
the code only has 1 goal. 

The code will roll a number between 0-99 inclusive any number between 0-9 inclusive will make the
tile have a thief and any number between 10-29 will make the tile a gold tile, this assumes that 
gold, or a thief cannot exist on the same tile. The goal tile can have one of these contents.

The maze has a method which returns the player's location as a string with co-ordinates. The maze 
can also return the gold amount the player currently has.

The driver class which is a command line project takes several arguments.
Here are the following **required** arguments and what they do.
```
-rows This takes the integer number above 0 after it as the number of rows

-cols This takes the integer number above 0 after it as the number of columns

-goldValue takes the integer number above 0 after it as the value pof the gold tiles

-playerX The starting x position of the player

-playerY the starting y position of the player

-goalX the goal x position

-goalY the goal y position
```

The following are the optional arguments for the driver

```
-imperfect takes the integer value above 0 and below numberOfEdges - n + 1 where n is equal to the 
the number of cells or (rows * cols)

-wrapping takes no value after it but will make the maze a wrapping maze which means if you are at 
(0, 0) and you go north you will be at (0, rows-1)

-seed this takes a integer value and sets it as the seed for the maze generation and ranomization 
portions of the project
```

An example game is as follows
```
$ java -jar HW03-Maze.jar -rows 3 -cols 3 -goldValue 10 -playerX 0 -playerY 0 -goalX 2 -goalY 2 -seed 555
Welcome to the maze you can move around by typing the first character of the cardinal directions (N, E, S, W).
You are currently at tile (0, 0), your gold is 0, your possible moves are:[SOUTH, EAST]
s
You are currently at tile (0, 1), your gold is 0, your possible moves are:[NORTH, SOUTH]
s
You are currently at tile (0, 2), your gold is 0, your possible moves are:[NORTH]
N
You are currently at tile (0, 1), your gold is 0, your possible moves are:[NORTH, SOUTH]
N
You are currently at tile (0, 0), your gold is 0, your possible moves are:[SOUTH, EAST]
e
You are currently at tile (1, 0), your gold is 0, your possible moves are:[SOUTH, EAST, WEST]
E
The location has gold of value: 10
You are currently at tile (2, 0), your gold is 10, your possible moves are:[SOUTH, WEST]
S
You are currently at tile (2, 1), your gold is 10, your possible moves are:[NORTH]
n
You are currently at tile (2, 0), your gold is 10, your possible moves are:[SOUTH, WEST]
w
You are currently at tile (1, 0), your gold is 10, your possible moves are:[SOUTH, EAST, WEST]
s
You are currently at tile (1, 1), your gold is 10, your possible moves are:[NORTH, SOUTH]
s
You are currently at tile (1, 2), your gold is 10, your possible moves are:[NORTH, EAST]
e
You Won, you completed the maze your final gold was: 10
```