import java.util.Arrays;

/**
 * @author Steiners
 *
 */

public final class ArrayBag<T> implements BagInterface<T>{

	private final T[] bag;
	private int numberOfEntries;
	private static final int DEFAULT_CAPACITY=25;
	
	/** Creates an empty bag whose initial capacity is 25. */
	public ArrayBag()
	{
		this (DEFAULT_CAPACITY);
	} // end default constructor
	
	
	/** 
	 * Creates an empty bag having a given capacity.
     * @param desiredCapacity  The integer capacity desired.
     */
	public ArrayBag(int desiredCapacity)
    {
		bag=(T[]) new Object[desiredCapacity];
		numberOfEntries=0;
	} // end constructor

	
	
	@Override
    /** Returns the value of the numberOfEntries which tracks the number of items within a bag */
	public int getCurrentSize() {return numberOfEntries;}

	@Override
    /** Checks if the bag is empty by testing the numberOfEntries */
	public boolean isEmpty() {return numberOfEntries <= 0;}

	@Override
    /** Adds a newEntry to the end of the bag then increases the numberOfEntries by one */
	public boolean add(T newEntry) {
		try{
			bag[numberOfEntries] = newEntry;
			numberOfEntries ++;
			return true;
		}
		catch (ArrayIndexOutOfBoundsException e){
			return false;
		}
	}

	@Override
    /**
     * Removes the last entry unless the bag is empty
     * @return the value of the removed item unless empty
     */
	public T remove() {
	    if(isEmpty()){
	        return null;
        }
        else{
	        numberOfEntries --;
	        return bag[numberOfEntries];
        }
	}

	@Override
    /**
     * Removes all instances of anEntry
     * @return true if an item was removed false if not
     */
	public boolean remove(T anEntry) {
		boolean removed = false;
		for(int i = numberOfEntries -1; i >= 0; i --){
		    if(bag[i] == anEntry) {
                if( i == numberOfEntries-1){
                    numberOfEntries --;
                }
                else{
                    bag[i] = bag[numberOfEntries -1];
                    numberOfEntries --;
                }
                removed = true;
            }
		}
		return  removed;
	}

	@Override
    /** Empties the bag by setting the numberOfEntries to 0*/
	public void clear() { numberOfEntries = 0;}

	@Override
    /** gets the number of times anEntry appears in the bag*/
	public int getFrequencyOf(T anEntry) {
		int frequency = 0;
		for( int i = 0; i < numberOfEntries; i++){
			if(bag[i] == anEntry){
				frequency ++;
			}
		}
		return frequency;
	}

	@Override
    /** @returns True if anEntry is inside the bag false if not */
	public boolean contains(T anEntry) {
		for( int i = 0; i < numberOfEntries; i++){
			if(bag[i] == anEntry){
				return true;
			}
		}
		return false;
	}

	@Override
    /** Returns an array whose content is just that of the bag */
	public T[] toArray() {
		return Arrays.copyOfRange(bag, 0, numberOfEntries);
	}
	
}
