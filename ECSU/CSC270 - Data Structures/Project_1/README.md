# Project 1: Java Primers and Object-Oriented Design

After completing the project students will learn:
* to create classes in Java
* to define class member functions
* to write a testing program
* to read and write files in Java
* to debug Java code
* software development phases (design, actual coding, documentation, testing and debugging)

**Question 1**

Create a class called Complex for performing arithmetic with complex numbers. Use the given testing
program to test your class. Complex numbers have the form realPart + imaginaryPart * i where i is (-
1)^(½).

Use integers to represent the realPart and imaginaryPart components as the private members of the
class. Provide a constructor function that enables an object of this class initialized when it is declared.
The constructor should contain default values in case no initial values provided. Provide public member
functions for each of the following:

```
a) Addition of two Complex numbers: The real parts added together and the imaginary parts added
together.
b) Subtraction of two Complex numbers: The real part of the right operand is subtracted from the
real part of the left operand and the imaginary part of the right operand is subtracted from the
imaginary part of the left operand.
c) Printing Complex numbers in the form (a, b) where a is the real part and b is the imaginary
part.
```
The testing program is given as follows:

```
public static void main(String[]args)
{
Complex c1= new Complex(3, -5);
Complex c2= new Complex(8, 7);
Complex sum = new Complex();
sum.add(c1, c2);
Complex diff = new Complex();
diff.subtract(c1, c2);
System. _out_ .println("The sum is ");
Complex. _printResult_ (sum);
System. _out_ .println("The difference is ");
Complex. _printResult_ (diff);

}
```

**Question 2**

Write a grading program for a class with the following grading policies:

1. There are two quizzes, each graded on the basis of 10 points.
2. There is one midterm exam and one final exam, each graded on the basis of 100 points.
3. The final exam counts for 50% of the grade, the midterm counts for 25%, and two quizzes each
    count for 12.5%. (Do not forget to normalize the quiz scores. They should be converted to a
    percent before they are averaged in.)

Any grades of 90 or more is an A, any grades of 80 or more (but less than 90) is a B, any grades of 70 or
more (but less than 80) is a C, any grades of 60 or more (but less than 70) is a D, and any grades below
60 is an F.

The program will read in students’ scores from an input file named “input.txt” and output the students’
records to a file named “output.txt”, which consist of student’s ID, name, two quizzes and two exam
scores as well as the students’ average score for the course and final letter grade. Define and use a class
for a student record. **Note: input/output should be done with files.**

The organization for the input file is as follows: The number at the top represents the number of
students in a class. Following are students’ records, which consist of 6 items:

* student ID
* name
* quiz 1
* quiz 2
* midterm exam
* final exam

“input.txt”
```
10
0001 bucky   10 10 70 90  
0002 joe     7 9 90 100
0003 mike    8 8 75 70 
0004 nick    10 7 90 80
0005 jenn    9 9 100 95
0006 elsa    8 9 70 85
0007 jack    9 10 85 74
0008 will    5 7 75 60
0009 seth    9 7 90 85
0010 rose    5 5 65 60
```

After running your program, the system should generate a file named “output.txt” which looks like

```
ID	Name	Quiz1	Quiz2	Midterm	Final	Avg.	Grade
0001	bucky	10	10	70	90	87.5	B
0002	joe	7	9	90	100	92.5	A
0003	mike	8	8	75	70	73.75	C
0004	nick	10	7	90	80	83.75	B
0005	jenn	9	9	100	95	95.0	A
0006	elsa	8	9	70	85	81.25	B
0007	jack	9	10	85	74	82.0	B
0008	will	5	7	75	60	63.75	D
0009	seth	9	7	90	85	85.0	B
0010	rose	5	5	65	60	58.75	F
```
