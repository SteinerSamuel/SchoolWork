# Homework 1 - Birds

This is an implementation of the problem statement from homework 1.

## How To Use
Almost all functionality can be accessed from the ConservatoryConcrete class which is an 
implementation of the Conservatory interface.

Of the functionality listed in the homework everything is working.


    Allow you to rescue new birds and bring them into your conservatory.

    Calculate what food needs to be kept and in what quantities.

    Assign a bird to a given aviary in the conservatory. Assignments must follow the following criteria:

        There is a maximum of 20 aviaries in the conservatory.

        Any bird can be inserted into an empty aviary.

        No aviary can house more than 5 birds.

        No extinct birds can be added to an aviary.

        Flightless birds, birds of prey, and waterfowl should not be mixed with other bird types.

    Allow a guest to look up which aviary a bird is in.

    Produce text for any given aviary that gives a description of the birds it houses and any interesting information that it may have about that animal.

    Produce a “directory” that lists all the aviaries by location and the birds they house.

    Produce an index that lists all birds in the conservatory in alphabetical order and their location.
Lets start by making a conservatory:

```Java
Conservatory conservatory = new ConservatoryConcrete();
```

To add a bird you make a Bird object using the following code as an example this creates a new hawk:
```Java
Bird bird= new Hawk("Henry");
```
replace the Hawk with any bird type which is implemented and a bird will be created.
If a bird needs to bypass the classification you can call the classes above it such as ```BirdsOfPray```

Once the bird has been made you can then add it to a conservatory using the ```addBird``` method. 
The conservatory will already need to have an aviary. If it does not you can add one using the 
```addAviary ``` method as follows.

```Java
Aviary aviary1 = conservatory.addAviary("Location");
```
This will add an aviary to the conservatory and return the aviary which you can use later to specify
which aviary you want to add the bird to.

Now to add the bird.

```Java
conservatory.addBird(bird, aviary1);
``` 

This will add the bird to the specified aviary. This method makes the correct checks to make sure,
the birds which will be added meets the criteria. The add bird method which is directly part of the 
aviary class does NOT do these checks.

You can also calculate the food needed for all birds with the ```calcFood``` method. This method 
returns a string which showcases the necessary amount of food for the birds in the conservatory.

Each method is documented within the code with JavaDoc formatting.

## What Could be Added
Something which could be implemented which wasn't, is the ability to remove a bird or aviary from the
conservatory, this could even try to find the birds new homes within the conservatory. 

There could also be more automation within the ```addBird``` method, by maybe trying to find an 
aviary which the bird could be added. This could remove the need to provide the aviary.

## Design Changes
There were plenty of design changes from the original design document, the biggest change is using
more interfaces and implementing 'square' design to extend interfaces with the ```talkingBird``` and
 ```waterBird```. This allowed for the program to follow basic principles of the interface promise
 and makes sure no classes have methods which are not within an interface. 
 
 Most of the other changes came from just missing methods which I did not add to the original design 
 document.
 
 The only change which does not fall under these two categories was the changing of bodyOfWater from
 an Enum. I decided to make this change to allow the class to implement how they wanted to represent
 the bodyOfWater this way there could be any type of body of water which I as the developer did not
 think of.