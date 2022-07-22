#include <ctype.h>
#include <stdlib.h>
#include "caeser.h"

void encode(char *plaintext, int key)
{
	char *p = malloc(sizeof(char));
	p = plaintext;
	while (*plaintext) {
		// Make the character upper case
		*p = toupper(*plaintext);
		// Check to see if its an encodable character
		if (*plaintext >= 'A' && *plaintext <= 'Z') {
			//  if the character wraps back to the start of the alphabet, what to do
			if (*plaintext >=
			    'Z' - (char)key % 26 +
				    1) //The add one is to deal with the fact we start at Z
			{
				*p = 'A' +
				     (char)(key % 26 - ('Z' - *plaintext) -
					    1); // The subtract one is to deal with the fact we start at the wrap around
			}
			// In all other cases just add
			else {
				*p = *plaintext + (char)key % 26;
			}
		}
		plaintext++;
		p++;
	}
}

void decode(char *plaintext, int key)
{
	while (*plaintext) {
		// Make the character upper case
		*plaintext = toupper(*plaintext);
		// Check to see if its an encodable character
		if (*plaintext >= 'A' && *plaintext <= 'Z') {
			//  if the character wraps to the end of the alphabet, what to do
			if (*plaintext <=
			    'A' + (char)key % 26 -
				    1) //The minus one is to deal with the fact we start at A
			{
				*plaintext =
					'Z' -
					(char)(key % 26 - (*plaintext - 'A') -
					       1); // The subtract one is to deal with the fact we start at the wrap around
			}
			// In all other cases just add
			else {
				*plaintext = *plaintext - (char)key % 26;
			}
		}
		plaintext++;
	}
}
