#include <stdio.h>
#include "caeser.h"

int main()
{
	char greeting[] = "Hello world!";

	printf("Simple Test: String Hello World! Key: 5\n");
	printf("%s\n", greeting);
	encode(greeting, 5);
	printf("%s\n", greeting);
	decode(greeting, 5);
	printf("%s\n", greeting);

	printf("Simple Test: String Hello World! Key: 31, should be same as above\n");
	printf("%s\n", greeting);
	encode(greeting, 31);
	printf("%s\n", greeting);
	decode(greeting, 31);
	printf("%s\n", greeting);

	printf("Negative key Test: String Hello World! Key: -3,\n");
	printf("%s\n", greeting);
	encode(greeting, -3);
	printf("%s\n", greeting);
	decode(greeting, -3);
	printf("%s\n", greeting);

	char greeting2[] = "The quick brown fox jumps over the lazy dog";
	printf("Pangram test key: 6\n");
	printf("%s\n", greeting2);
	encode(greeting2, 6);
	printf("%s\n", greeting2);
	decode(greeting2, 6);
	printf("%s\n", greeting2);
}