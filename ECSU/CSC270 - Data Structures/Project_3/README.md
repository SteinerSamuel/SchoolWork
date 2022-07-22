
# Project 3: Linked Based Bag

Write a class LinkedBag, which includes the following operations (the same as the ArrayBag in Project 2)

    •Get the current number of entries in this bag.
    •See whether this bag is empty.
    •Add a new entry to this bag.
    •Remove one unspecified entry from this bag, if possible.
    •Remove all occurrences of a given entry from this bag.
    •Remove all entries from this bag.
    •Count the number of times a given entry appears in this bag.
    •Test whether this bag contains a given entry.
    •Retrieve all entries that are in this bag.

The bag specification "BagInterface.java" is the same file as we provided for Project 2. The test driver "LinkedBagTest.java" is quite similar to the test driver given in Project 2 except minor change for two places. Please read them carefully and complete your class implementation "LinkedBag.java".  The output should be look like 

```
Testing an initially empty bag:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Testing isEmpty with an empty bag:
isEmpty finds the bag empty: OK.


Testing the method getFrequencyOf:
In this bag, the count of Two is 0

Testing the method contains:
Does this bag contain Two? false

Removing a string from the bag:
remove() returns null
The bag contains 0 string(s), as follows:


Removing "Two" from the bag:
remove("Two") returns false
The bag contains 0 string(s), as follows:

+++++++++++++++++++++++++++++++++++++++++++++++++++

Adding 6 strings to an initially empty bag with the capacity to hold more than 6 strings:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adding One One Two One Three One
The bag contains 6 string(s), as follows:
One Three One Two One One
Testing isEmpty with a bag that is not empty:
isEmpty finds the bag not empty: OK.


Testing the method getFrequencyOf:
In this bag, the count of One is 4
In this bag, the count of Two is 1
In this bag, the count of Three is 1
In this bag, the count of Four is 0
In this bag, the count of XXX is 0

Testing the method contains:
Does this bag contain One? true
Does this bag contain Two? true
Does this bag contain Three? true
Does this bag contain Four? false
Does this bag contain XXX? false

Removing a string from the bag:
remove() returns One
The bag contains 5 string(s), as follows:
Three One Two One One

Removing "Two" from the bag:
remove("Two") returns true
The bag contains 4 string(s), as follows:
One Three One One

Removing "One" from the bag:
remove("One") returns true
The bag contains 1 string(s), as follows:
Three

Removing "Three" from the bag:
remove("Three") returns true
The bag contains 0 string(s), as follows:


Removing "XXX" from the bag:
remove("XXX") returns false
The bag contains 0 string(s), as follows:


Clearing the bag:
Testing isEmpty with an empty bag:
isEmpty finds the bag empty: OK.

The bag contains 0 string(s), as follows:

+++++++++++++++++++++++++++++++++++++++++++++++++++


Testing an initially empty bag that  will be filled to capacity:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adding One Two One Three Two Three Four
The bag contains 7 string(s), as follows:
Four Three Two Three One Two One
Try to add XXX to the full bag:
Added a string beyond the bag's capacity: OK!
The bag contains 8 string(s), as follows:
XXX Four Three Two Three One Two One 
```
