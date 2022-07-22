#include <stdio.h>
#include <string.h>
#include "caeser.h"
#include "queue.h"

void print_process(process_t process)
{
	printf("[id: %d, name: %s]", process.identifier, process.name);
}

void print_list(queue_t *queue)
{
	if (queue->front == NULL) {
		printf("[Empty]\n");
	} else {
		struct node *current = queue->front;
		while (current->next) {
			print_process(*((process_t *)current->data));
			current = current->next;
			printf(" -> ");
		}
		print_process(*((process_t *)current->data));
		printf("\n");
	}
}

int main()
{
	int key = 3;
	queue_t q = create_queue();
	queue_t *queue = &q;

	char nameA[] = "Hello World";
	char nameB[] = "Ceaser";
	char nameC[] = "Cipher";
	char nameD[] = "Queue";

	process_t process1 = create_process(1, nameA);

	process_t process2 = create_process(2, nameB);

	process_t process3 = create_process(3, nameC);

	process_t process4 = create_process(4, nameD);

	printf("Enqueue: ");
	print_process(process1);
	encode(process1.name, key);
	enqueue(queue, &process1);
	printf(" is enqueued. ");
	print_list(queue);

	printf("Enqueue: ");
	print_process(process2);
	encode(process2.name, key);
	enqueue(queue, &process2);
	printf(" is enqueued. ");
	print_list(queue);

	printf("Enqueue: ");
	print_process(process3);
	encode(process3.name, key);
	enqueue(queue, &process3);
	printf(" is enqueued. ");
	print_list(queue);

	printf("Dequeue: ");
	process_t *dequed = (process_t *)dequeue(queue);
	decode(dequed->name, key);
	print_process(*dequed);
	printf(" is dequeued. ");
	print_list(queue);

	printf("Enqueue: ");
	print_process(process4);
	encode(process4.name, key);
	enqueue(queue, &process4);
	printf(" is enqueued. ");
	print_list(queue);

	printf("Dequeue: ");
	dequed = (process_t *)dequeue(queue);
	decode(dequed->name, key);
	print_process(*dequed);
	printf(" is dequeued. ");
	print_list(queue);

	printf("Dequeue: ");
	dequed = (process_t *)dequeue(queue);
	decode(dequed->name, key);
	print_process(*dequed);
	printf(" is dequeued. ");
	print_list(queue);

	printf("Enqueue: ");
	print_process(process1);
	encode(process1.name, key);
	enqueue(queue, &process1);
	printf(" is enqueued. ");
	print_list(queue);

	printf("Dequeue: ");
	dequed = (process_t *)dequeue(queue);
	decode(dequed->name, key);
	print_process(*dequed);
	printf(" is dequeued. ");
	print_list(queue);
}