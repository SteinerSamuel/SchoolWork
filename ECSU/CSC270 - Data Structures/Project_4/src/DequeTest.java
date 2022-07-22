public class DequeTest {

	public static void display(CArrayDeque dq) {
		System.out.println("The items in queue:");
		for (int i=0; i<dq.toArray().length; i++) {
			System.out.print(dq.toArray()[i]+" ");
		}
		System.out.println();
	}
	
	public static <T> void checkFrontBack(CArrayDeque dq) {

		T frontItem, backItem;
		frontItem=(T) dq.retrieveFront();
		backItem=(T) dq.retrieveBack();
		System.out.println("The front entry is " + frontItem);
		System.out.println("The back entry is " + backItem);
	}
	
		
	public static <T> void main(String[] args) {
	
		CArrayDeque<String> q =new CArrayDeque<>();
		System.out.println("The queue is empty now.");
		display(q);
		checkFrontBack(q);

		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		System.out.println("After adding at FRONT with One, Two, Three, Four, Five ...");
		
		q.addFront("One");
		q.addFront("Two");
		q.addFront("Three");
		q.addFront("Four");
		q.addFront("Five");
		
		display(q);
		checkFrontBack(q);
		
		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		System.out.println("After adding at BACK with 1, 2, 3, 4, 5 ...");
		
		q.addBack("1");
		q.addBack("2");
		q.addBack("3");
		q.addBack("4");
		q.addBack("5");
		display(q);
		checkFrontBack(q);

		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		System.out.println("Try to add at BACK with 6 ...");
		boolean flag=q.addBack("6");
		if (flag==true)
			System.out.println("Insert successfully!");
		else
			System.out.println("Insert failed!  OVERFLOW !");
		
		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		System.out.println("Delete 6 entries at FRONT  ...");
		for (int i=0; i<6; i++)		{
			flag=q.removeFront();	
			if (flag==true)
				System.out.println("Delete successfully!");
			else
				System.out.println("Insert failed!");
		}
		display(q);
		checkFrontBack(q);

		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		System.out.println("Delete 1 entry at Back and 1 entry at FRONT ...");
		flag=q.removeBack();	
		if (flag==true)
			System.out.println("Delete successfully!");
		else
			System.out.println("Insert failed!");
		flag=q.removeFront();	
		if (flag==true)
			System.out.println("Delete successfully!");
		else
			System.out.println("Insert failed!");
		display(q);
		checkFrontBack(q);

		
		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		System.out.println("Continue to delete 6 entries at FRONT  ...");
		
		for (int i=0; i<6; i++)		{
			flag=q.removeFront();	
			if (flag==true)
				System.out.println("Delete successfully!");
			else
				System.out.println("Empty Queue! Delete failed! UNDERFLOW!");
		}
		
		display(q);
		checkFrontBack(q);

	}

}
