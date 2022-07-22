
public class Complex {
	
	private int real;
	private int imaginary;
	
	Complex(){
		real=0;
		imaginary=0;
	}
	
	Complex(int x, int y){
		real=x;
		imaginary=y;
	}
	
	public Complex add (Complex x, Complex y) {
		real=x.real+y.real;
		imaginary=x.imaginary+y.imaginary;
		return this;
	}
	
	public Complex diff (Complex x, Complex y) {
		real=x.real-y.real;
		imaginary=x.imaginary-y.imaginary;
		return this;
	}

	
	public void display () {

		System.out.println("("+real+","+imaginary+")");
		
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		Complex num1=new Complex(5, 8);
		Complex num2=new Complex(2, -9);
		
		Complex num3=new Complex();
		num3.display();
		
		num3.add(num1, num2);
		num3.display();
		
		Complex num4 =new Complex();
		num4.display();
		
		num4.diff(num1, num2);
		num4.display();
		
	}

}
