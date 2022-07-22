package sorting;
import java.util.*;

/**
 * This program tests some of the functionality of the ComparisonSort class. 
 * It does not completely test the ComparisonSort class.  You should make sure 
 * that you do completely test your ComparisonSort class, either by modifying 
 * this file or by writing a different driver.
 */
public class TestSort {

    /**
     * Main method to run the ComparisonSort class.
     * 
     * @param args  a two-value array: first the number of items in the input
     *              array, then the random number seed (integer)to use in
     *              generating values
     */
	
	
	public static <T extends SortObject> void help()
	{
	   System.out.println("User options are:\n");
	   System.out.println("[H]elp  [Q]uit  write [D]ata write sorted [O]utput");
	   System.out.println("[R]un all sorting algorithms and get comparison results.");
	  
	   System.out.println("[0] insertion sort");
	   System.out.println("[1] selection sort");
	   System.out.println("[2] shell sort");
	   System.out.println("[3] quick sort");
	   System.out.println("[4] heap sort");
	   System.out.println("[5] bubble sort ");
	   System.out.println("[6] merge sort ");
	   System.out.println();
	}

	public static <T extends SortObject> void intro()
	{
		System.out.println("Testing program for sorting methods for a contiguous list.");
		help ();
	}

	public static <T extends SortObject> void traverse(T[] A)
	{
		int counter=0;
		for (int i=0; i<A.length-1; i++) {
			counter++;
			System.out.print(A[i]+"," );
			if (counter==50)
			{
				System.out.println();
				counter=0;
			}
		}
		System.out.println(A[A.length-1]+"\n" );


	}

	@SuppressWarnings("unchecked")
	public static <T extends SortObject> T[] makeCopy(T[] a, int size)
	{
		 SortObject[] copy_arr = new SortObject[size];
	       for (int k = 0; k < size; k++) {
	    	     copy_arr[k]=new SortObject(a[k].getData());
	       }
	       return (T[])copy_arr;

	}
    public static void main(String[] args) {

        if (args.length != 2) {
            System.err.println("Expected 2 but got " + args.length);
            System.err.println("Arguments expected:");
            System.err.println("  # items in input array");
            System.err.println("  random # seed");
            System.exit(1);
            }
        int arrSize = Integer.parseInt(args[0]);
        int seed = Integer.parseInt(args[1]);
        
        intro();

        // Create the input array of unsorted objects.
        SortObject[] arr = new SortObject[arrSize];

       // It is important to give the seed so you can reproduce results.
       Random random = new Random(seed);
       for (int k = 0; k < arrSize; k++) {
           arr[k] = new SortObject(random.nextInt(arrSize));
       }

       SortObject[] copy_arr = makeCopy(arr, arrSize); 
      
       
        System.out.println("Parameters used:");
        System.out.println("  # items in input array: " + arrSize);
        System.out.println("  random # seed: " + seed);

        String command = null;

        boolean flag=true;
 	   	while (flag) {
 	 	   		System.out.println("Enter a command of H, Q, O, D," ); 
 	   		System.out.println("0, 1, 2, 3, 4, 5, 6: "); 
 	   		command=new Scanner (System.in).nextLine(); 
 	      
 	   		switch (command) {
 	   		case "h":
 	   		case "H":
 	   			help();
 	   			break;

 	        case "d": 
 	        case "D":
 	            System.out.println("\nUnsorted list ");
 	            traverse(arr);
 	            System.out.println();
 	            break;

 	        case "o": 
 	        case "O":
 	            System.out.println("\nsorted list ");
 	            traverse(copy_arr);
 	            System.out.println();
 	            break;

 	        case "r": 
 	        case "R":
 	        	copy_arr = makeCopy(arr, arrSize);       	
            	SortingComparisons.runAllSorts(copy_arr);
 	        	System.out.println();
 	        	break;
 	        case "q": 
 	        case "Q":
 	        	flag=false;
 	        	break;
 	        	
 	       
 	        case "0": case "1": case "2": case "3": case "4": case "5": case "6": 
	        {
	           
	        	SortObject.resetCompares();
	        	SortObject.resetAssignments();
	        	//           Timer clock = null;
	           
	        	long startTime=0, endTime=0;
	        	switch (command) {
	        	case "0":
	        		copy_arr = makeCopy(arr, arrSize);  
	        		System.out.println("Insertion Sort ");
	        		startTime = System.currentTimeMillis();
	            	SortingComparisons.insertionSort(copy_arr);
	            	endTime = System.currentTimeMillis();
	            	break;
	              
	        	case "1": 
	        		copy_arr = makeCopy(arr, arrSize);       	
	        		System.out.println("Selection Sort ");
	        		startTime = System.currentTimeMillis();
	        		SortingComparisons.selectionSort(copy_arr);
	            	endTime = System.currentTimeMillis();
	            	break;
	              
	        	case "2": 
	        		copy_arr = makeCopy(arr, arrSize);       	
	        		System.out.println("    Shell Sort ");
	        		startTime = System.currentTimeMillis();
	        		SortingComparisons.shellSort(copy_arr);
	            	endTime = System.currentTimeMillis();
	            	break;
              
	        	case "3": 
	        		copy_arr = makeCopy(arr, arrSize); 
	        		System.out.println("    Quick Sort ");
	        		startTime = System.currentTimeMillis();
	        		SortingComparisons.quickSort(copy_arr);
	            	endTime = System.currentTimeMillis();
	            	break;
	              
	        	case "4": 
              	  	copy_arr = makeCopy(arr, arrSize); 
              	  	System.out.println("     Heap Sort ");
              	  	startTime = System.currentTimeMillis();
              	  	SortingComparisons.heapSort(copy_arr);
	            	endTime = System.currentTimeMillis();
	            	break;
	          
	        	case "5": 
	        		copy_arr = makeCopy(arr, arrSize); 
	        		System.out.println("     Bubble Sort ");
	        		startTime = System.currentTimeMillis();
	        		SortingComparisons.bubbleSort(copy_arr);
	            	endTime = System.currentTimeMillis();
	            	break;
			  
	        	case "6": 
	        		copy_arr = makeCopy(arr, arrSize); 
	        		System.out.println("     Merge Sort ");
	        		startTime = System.currentTimeMillis();
	        		SortingComparisons.mergeSort(copy_arr);
	            	endTime = System.currentTimeMillis();
	            	break;
	          
	        	}
	        	
	        	System.out.println("Comparisons: " + SortObject.getCompares() );
	        	System.out.println("Assignments: " + SortObject.getAssignments());
	        	System.out.println("Time used: " + (endTime -startTime)+ " milliseconds.");
	        	System.out.println();
	        	System.out.println();
	           
	         }
	         break;	
 	        	
 	        	default:
 	        		break;
 		   }
 
 	   	}
 	  	
 	  	System.exit(1);
    }
}
