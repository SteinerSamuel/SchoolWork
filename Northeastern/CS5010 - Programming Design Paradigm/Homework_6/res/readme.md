# HUNT THE WUMPUS
___
This is an mvc version of hunt the wumpus.

To run this all you need to do is traverse to the res file and run the jar
using one of the following commands.

```
java -jar Homework_6.jar --gui
```

or 

```
java -jar Homework_6.jar --gui
```
___
## Design changes
___
From the last homework I had to totally redesign my controller, as it was designed in the wrong way.
Now the controller handles user input validation as well as doesn't just make model public methods, 
there are no longer any methods which the view pulls data from the controller, instead the data is 
either pushed by the controller or gotten directly from the model.
