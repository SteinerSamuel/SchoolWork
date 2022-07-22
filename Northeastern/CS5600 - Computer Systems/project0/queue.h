#ifndef QUEUE_H_INCLUDED
#define QUEUE_H_INCLUDED

typedef struct process_t {
	int identifier;
	char *name;
} process_t;

struct node {
	void *data;
	struct node *next;
};

typedef struct queue_t {
	struct node *front;
	struct node *rear;
} queue_t;

queue_t create_queue();

process_t create_process(int id, char *name);

void enqueue(queue_t *queue, void *element);

void *dequeue(queue_t *queue);

#endif /* QUEUE_H_INCLUDED */