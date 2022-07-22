CC = gcc

main:
	$(CC) -o caeser caeser_main.c caeser.c
	$(CC) -o queue queue_main.c queue.c
	$(CC) -o ciphered_queue ciphered_queue_main.c queue.c caeser.c
	
caeser:
	$(CC) -o caeser caeser_main.c caeser.c

queue:
	$(CC) -o queue queue_main.c queue.c

ciphered_queue:
	$(CC) -o ciphered_queue ciphered_queue_main.c queue.c caeser.c