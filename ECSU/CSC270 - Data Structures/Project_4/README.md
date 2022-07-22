# Project 4: Circular Array Based Deque

Using a circular array to implement a deque, which includes the following operations

 * create an empty deque
 * test if the deque is empty
 * insert a new item at front into the deque
 * insert a new item at back into the deque
 * remove the item at front in the deque
 * remove the item at back in the deque
 * get the item at front in the deque
 * get the item at back in the deque
 * Retrieves all entries that are in the deque

The framework of the CArrayDeque class is given in "CArrayDeque.java" and a test driver is given in "DequeTest.java". Please read them carefully and complete your class methods in "CArrayDeque.java". Following is an example of the output.
```
The queue is empty now.
The items in queue:
null null null null null null null null null null
The front entry is null
The back entry is null
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After adding at FRONT with One, Two, Three, Four, Five ...
The items in queue:
null null null null null Five Four Three Two One
The front entry is Five
The back entry is One
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After adding at BACK with 1, 2, 3, 4, 5 ...
The items in queue:
1 2 3 4 5 Five Four Three Two One
The front entry is Five
The back entry is 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Try to add at BACK with 6 ...
Insert failed!  OVERFLOW !
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Delete 6 entries at FRONT  ...
Delete successfully!
Delete successfully!
Delete successfully!
Delete successfully!
Delete successfully!
Delete successfully!
The items in queue:
null 2 3 4 5 null null null null null
The front entry is 2
The back entry is 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Delete 1 entry at Back and 1 entry at FRONT ...
Delete successfully!
Delete successfully!
The items in queue:
null null 3 4 null null null null null null
The front entry is 3
The back entry is 4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Continue to delete 6 entries at FRONT  ...
Delete successfully!
Delete successfully!
Empty Queue! Delete failed! UNDERFLOW!
Empty Queue! Delete failed! UNDERFLOW!
Empty Queue! Delete failed! UNDERFLOW!
Empty Queue! Delete failed! UNDERFLOW!
The items in queue:
null null null null null null null null null null
The front entry is null
The back entry is null
```
