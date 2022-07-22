package sorting;

/**
 * This class implements six different comparison sorts (and an optional
 * heap sort for extra credit):
 * 
 * insertion sort
 * selection sort
 * shell sort
 * quick sort
 * bubble sort
 * merge sort
 * (extra credit) heap sort
 * 
 * It also has a method that runs all the sorts on the same input array and
 * prints out statistics.
 */

public class SortingComparisons{

    /**
     * Sorts the given array using the insertion sort algorithm. Note: after
     * this method finishes the array is in sorted order.
     * 
     * @param <T>  the type of values to be sorted
     * @param A    the array to sort
     */
    public static <T extends SortObject> void insertionSort(T[] A) {
        for (int i=1; i<A.length; i++)
        {
            for (int j = i; j > 0; j --){
                if (A[j].compareTo(A[j-1]) == -1){
                    int temp = A[j].getData();
                    A[j].setData(A[j-1].getData());
                    A[j-1].setData(temp);
                }
            }
        }
    }


    /**
     * Sorts the given array using the selection sort algorithm. 
     * Note: after this method finishes the array is in sorted order.
     * 
     * @param <T>  the type of values to be sorted
     * @param A    the array to sort
     */
    public static <T extends SortObject> void selectionSort(T[] A) {
        for(int i = 0; i < A.length-1; i++){
            int place = i;
            for(int b = i + 1; b < A.length; b++){
                if(A[b].compareTo(A[place]) == -1){
                    place = b;
                }
            }
            int temp = A[i].getData();
            A[i].setData(A[place].getData());
            A[place].setData(temp);
        }
    }  

    /**
     * Sorts the given array using the bubble sort algorithm.
                * Note: after this method finishes the array is in sorted order.
     *
     * @param <T>  the type of values to be sorted
                * @param A    the array to sort
     */
        public static <T extends SortObject> void bubbleSort(T[] A) {
            for(int i = 0; i < A.length; i++){
                for (int b = 0; b < A.length -1; b++){
                    if(A[b].compareTo(A[b+1]) == 1){
                        A[b].setData(A[b].getData() + A[b+1].getData());
                        A[b+1].setData(A[b].getData() - A[b+1].getData());
                        A[b].setData(A[b].getData() - A[b+1].getData());
                    }
                }
            }
        
    }
    
    
    /**
     * Sorts the given array using the shell sort algorithm.
     * Note: after this method finishes the array is in sorted order.
     * 
     * @param <T>  the type of values to be sorted
     * @param A    the array to sort
     */
 	public static <T extends SortObject> void shellSort(T[] A){
 	    int n = A.length;
 	    for (int gap = n/2; gap > 0; gap /=2){
 	        for(int i = gap; i < n; i ++){
 	            int temp = A[i].getData();

                int j;
 	            for (j =i; j>= gap && A[j-gap].compareTo(A[i]) == 1; j-= gap)
 	                A[j].setData(A[j-gap].getData());

                A[j].setData(temp);
            }
        }
 	} // end shellSort


    /**
     * Sorts the given array using the merge sort algorithm.
     * Note: after this method finishes the array is in sorted order.
     *
     * @param <T>  the type of values to be sorted
     * @param A    the array to sort
     */

    public static <T extends SortObject> void mergeSort(T[] A) {
        if(A == null)
        {
            return;
        }

        if(A.length > 1)
        {
            int mid = A.length / 2;

            // Split left part
            SortObject[] left = new SortObject[mid];
            for (int k = 0; k < mid; k++) {
                left[k]=new SortObject(A[k].getData());
            }

            // Split right part
            SortObject[] right = new SortObject[A.length-mid];
            for (int k = mid; k < A.length; k++) {
                right[k -mid ]=new SortObject(A[k].getData());
            }

            mergeSort(left);
            mergeSort(right);

            int i = 0;
            int j = 0;
            int k = 0;

            // Merge left and right arrays
            while(i < left.length && j < right.length)
            {
                if(left[i].compareTo(right[j]) < 0)
                {
                    A[k].setData(left[i].getData());
                    i++;
                }
                else
                {
                    A[k].setData(right[j].getData());
                    j++;
                }
                k++;
            }
            // Collect remaining elements
            while(i < left.length)
            {
                A[k].setData(left[i].getData());
                i++;
                k++;
            }
            while(j < right.length)
            {
                A[k].setData(right[j].getData());
                j++;
                k++;
            }
        }


    }



    /**
     * Sorts the given array using the quick sort algorithm, using the median of
     * the first, last, and middle values in each segment of the array as the
     * pivot value. 
     * Note: after this method finishes the array is in sorted order.
     * 
     * @param <T>  the type of values to be sorted
     * @param A   the array to sort
     */

    static <T extends SortObject> int partition(T[] A, int low, int high)
    {
        int pivot = A[high].getData();
        int i = (low-1); // index of smaller element
        for (int j=low; j<=high-1; j++)
        {
            // If current element is smaller than or
            // equal to pivot
            if (A[j].compareTo(A[high])<= 0)
            {
                i++;

                // swap arr[i] and arr[j]
                int temp = A[i].getData();
                A[i].setData(A[j].getData());
                A[j].setData(temp);
            }
        }

        // swap arr[i+1] and arr[high] (or pivot)
        int temp = A[i+1].getData();
        A[i+1].setData(A[high].getData());
        A[high].setData(temp);

        return i+1;
    }

    static <T extends SortObject> void qSort(T[] arr, int low, int high)
    {
        if (low < high)
        {
            /* pi is partitioning index, arr[pi] is
            now at right place */
            int pi = partition(arr, low, high);

            // Recursively sort elements before
            // partition and after partition
            qSort(arr, low, pi-1);
            qSort(arr, pi+1, high);
        }
    }

    public static <T extends SortObject> void quickSort(T[] A) {
        qSort(A, 0, A.length-1);
    }


        
    /**
     * Sorts the given array using the heap sort algorithm outlined below.
     * Note: after this method finishes the array is in sorted order.
     * 
     * The heap sort algorithm is:
     * 
     * for each i from 1 to the end of the array
     *     insert A[i] into the heap (contained in A[0]...A[i-1])
     *     
     * for each i from the end of the array up to 1
     *     remove the max element from the heap and put it in A[i]
     * 
     * 
     * @param <T>  the type of values to be sorted
     * @param A    the array to sort
     */

    static <T extends SortObject> void heapify(T[] arr, int n, int i)
    {
        int largest = i;  // Initialize largest as root
        int l = 2*i + 1;  // left = 2*i + 1
        int r = 2*i + 2;  // right = 2*i + 2

        // If left child is larger than root
        if (l < n && arr[l].compareTo(arr[largest]) >0)
            largest = l;

        // If right child is larger than largest so far
        if (r < n && arr[r].compareTo(arr[largest]) > 0)
            largest = r;

        // If largest is not root
        if (largest != i)
        {
            int swap = arr[i].getData();
            arr[i].setData(arr[largest].getData());
            arr[largest].setData(swap);

            // Recursively heapify the affected sub-tree
            heapify(arr, n, largest);
        }
    }

    public static <T extends SortObject> void heapSort(T[]  A)
    {
        int n = A.length;

        // Build heap (rearrange array)
        for (int i = n / 2 - 1; i >= 0; i--)
            heapify(A, n, i);

        // One by one extract an element from heap
        for (int i=n-1; i>=0; i--)
        {
            // Move current root to end
            int temp = A[0].getData();
            A[0].setData(A[i].getData());
            A[i].setData(temp);

            // call max heapify on the reduced heap
            heapify(A, i, 0);
        }
       	
    }    
    
       
    
 
    /**
     * Internal helper for printing rows of the output table.
     * 
     * @param sort          name of the sorting algorithm
     * @param compares      number of comparisons performed during sort
     * @param moves         number of data moves performed during sort
     * @param milliseconds  time taken to sort, in milliseconds
     */
    @SuppressWarnings("unused")
	private static void printStatistics(final String sort, final int compares, final int moves,
                                        final long milliseconds) {
        System.out.format("%-23s%,15d%,15d%,15d\n", sort, compares, moves, 
                          milliseconds);
    }

    /**
     * Sorts the given array using the six (heap sort with the extra credit)
     * different sorting algorithms and prints out statistics. The sorts 
     * performed are:
     * 
     * insertion sort
     * selection sort
     * shell sort
     * quick sort
     * bubble sort
     * merge sort
     * (extra credit) heap sort
     * 
     * The statistics displayed for each sort are: number of comparisons, 
     * number of data moves, and time (in milliseconds).
     * 
     * Note: each sort is given the same array (i.e., in the original order). 
     * 
     * @param A  the array to sort
     */
    public static <T extends SortObject>void runAllSorts(T[] A) {
        System.out.format("%-23s%15s%15s%15s\n", "algorithm", "data compares", 
                          "data moves", "milliseconds");
        System.out.format("%-23s%15s%15s%15s\n", "---------", "-------------", 
                          "----------", "------------");
        // TODO: run each sort and print statistics about what it did
        
        long startTime, endTime;
        T[] arr=reset(A);
        startTime = System.nanoTime();
        insertionSort(arr);
        endTime = System.nanoTime();

        System.out.format("%-23s%15s%15s%15s\n", "insertion sort", SortObject.getCompares(), 
                SortObject.getAssignments(),(endTime-startTime)/1000000);
        
        arr=reset(A);
        startTime = System.nanoTime();
        selectionSort(arr);
        endTime = System.nanoTime();
        System.out.format("%-23s%15s%15s%15s\n", "selection sort", SortObject.getCompares(), 
                 SortObject.getAssignments(),(endTime-startTime)/1000000);
  
        
        arr=reset(A);
        startTime = System.nanoTime();
        shellSort(arr);
        endTime = System.nanoTime();
        System.out.format("%-23s%15s%15s%15s\n", "shell sort", SortObject.getCompares(), 
                   SortObject.getAssignments(),(endTime-startTime)/1000000 );
          

        arr=reset(A);
        startTime = System.nanoTime();
        quickSort(arr);
        endTime = System.nanoTime();
          System.out.format("%-23s%15s%15s%15s\n", "quick sort", SortObject.getCompares(), 
                    SortObject.getAssignments(),(endTime-startTime)/1000000);
    

        arr=reset(A);
        startTime = System.nanoTime();
        heapSort(arr);
        endTime = System.nanoTime();
        System.out.format("%-23s%15s%15s%15s\n", "heap sort", SortObject.getCompares(), 
                    SortObject.getAssignments(),(endTime-startTime)/1000000);

          
        arr=reset(A);
        startTime = System.nanoTime();
        bubbleSort(arr);
        endTime = System.nanoTime();
        System.out.format("%-23s%15s%15s%15s\n", "bubble sort", SortObject.getCompares(), 
                    SortObject.getAssignments(),(endTime-startTime)/1000000);

           
        arr=reset(A);
        startTime = System.nanoTime();
        mergeSort(arr);
        endTime = System.nanoTime();
        System.out.format("%-23s%15s%15s%15s\n", "merge sort", SortObject.getCompares(), 
                    SortObject.getAssignments(),(endTime-startTime)/1000000);
    }
    
    @SuppressWarnings("unchecked")
	private static <T extends SortObject> T[] reset(T[] A){

    	SortObject[] arr = TestSort.makeCopy(A, A.length); 
        SortObject.resetCompares();
    	SortObject.resetAssignments();
    	return (T[])arr;
        
    }
}
