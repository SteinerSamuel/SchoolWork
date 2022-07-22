#include <stdlib.h>
#include <string.h>
#include "queue.h"

queue_t create_queue()
{
	queue_t q;
	q.front = q.rear = NULL;
	return q;
}

process_t create_process(int id, char *name)
{
	process_t process;
	process.identifier = id;
	process.name = name;
	// strcpy(process.name, name);
	return process;
}

struct node *create_node(void *data)
{
	struct node *n = (struct node *)malloc(sizeof(struct node));
	n->data = data;
	n->next = NULL;
	return n;
}

void enqueue(queue_t *queue, void *element)
{
	struct node *n = create_node(element);
	if (queue->front == NULL) {
		queue->front = n;
		queue->rear = n;
		return;
	} else {
		queue->rear->next = n;
		queue->rear = n;
	}
}

void *dequeue(queue_t *queue)
{
	if (queue->front == NULL)
		return NULL;

	void *temp = queue->front->data;

	queue->front = queue->front->next;

	if (queue->front == NULL)
		queue->rear = NULL;

	return temp;
}