/**
   A class of bags whose entries are stored in a chain of linked nodes.
	The bag is never full.
*/

public class LinkedBag<T> implements BagInterface<T>{


	private class Node 
	{
		private T    data; // Entry in bag
		private Node next; // Link to next node
	
		private Node(T dataPortion)
		{
			this(dataPortion, null);	
		} // end constructor
		
		private Node(T dataPortion, Node nextNode)
		{
			data = dataPortion;
			next = nextNode;	
		} // end constructor
	} // end Node

	
	private Node firstNode;       // Reference to first node
	private int numberOfEntries;

	// Default constructor
	public LinkedBag() 
	{
		firstNode=null;
		numberOfEntries= 0;
	} // end default constructor creates a empty
	
	@Override
	public int getCurrentSize() {
		return numberOfEntries;
	} // Returns the number of entries in the bag

	@Override
	public boolean isEmpty() {
		return numberOfEntries <= 0;
	} // Returns true if the bag is empty

	@Override
	public boolean add(T newEntry) {
		Node newNode = new Node(newEntry, firstNode);	
		firstNode = newNode;
		numberOfEntries ++;
		return true;
	} // adds a new entry to the bag at the beginning  

	@Override
	public T remove() {
		if(!isEmpty()){
			T temp = firstNode.data;
			firstNode = firstNode.next;
			numberOfEntries --;
			return temp;
		}
		else{
			return null;
		} // removes the first item in the bag (Linked List)
	}

 	// Locates a given entry within this bag.
	// Returns a reference to the node containing the entry, if located,
	// or null otherwise.
	private Node getReferenceTo(T anEntry)
	{
		Node currentNode; 
		
		for (currentNode = firstNode;currentNode != null; currentNode = currentNode.next)
		{
			if (anEntry==currentNode.data)
				return currentNode;
		} // end while
		return currentNode;
	} // end getReferenceTo

	//Removes all occurrences of anEntry from the bag
	@Override
	public boolean remove(T anEntry) {
        boolean removed = false;
        Node currentNode;
        Node previousNode = null;
        
        // walks through all entries of the bag
        for (currentNode = firstNode; currentNode != null; currentNode = currentNode.next){
            if (anEntry == currentNode.data) { 
            	// if the entry in the bag is the same as the specified value test to see if it is the first node
                if(previousNode == null){
                	// removes the first node and subtracts one from the numberOfEntries
                    firstNode = firstNode.next;
                    numberOfEntries --;
                    removed = true;
                }
                else{
                	// If its not the first node removes the node and fixes references 
                    previousNode.next = currentNode.next;
                    numberOfEntries --;
                    removed = true;
                }
            }
            else{
            	// if the entry is not removed sets the last node looked at as previousNode
            	previousNode = currentNode;
            }
        }
        return removed;
	}// Returns True if anEntry was found and removed

	@Override
	public void clear() {
		firstNode = null;
		numberOfEntries = 0;
	}// empties the bag completely setting numberOfEntries to 0

	@Override
	public int getFrequencyOf(T anEntry) {
	    int freq = 0;
		for (Node currentNode = firstNode; currentNode != null; currentNode = currentNode.next){
		    if (anEntry == currentNode.data){
		        freq ++;
            }
        }
		return freq;
	}// Gets the frequency of an entry 

	@Override
	public boolean contains(T anEntry) {
		if(getReferenceTo(anEntry)== null) {
            return false;
        }
        else {
		    return true;
        }
	}// checks if anEntry is in the bag

	@Override
	public T[] toArray() {
		@SuppressWarnings("unchecked")
		T[] result= (T[])new Object[numberOfEntries];
		int i =0;// index of array
		
		for(Node current = firstNode; current != null; current = current.next){
			result[i] = current.data;
			i ++;
        }
		return result;
	}// takes the bag and puts it into an array for easy printing
	
}
