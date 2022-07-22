#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>

#include "proj2.h"

struct queueNode
{
    int sock;
    struct queueNode *next;
};

char keys[MAX_KEY_NUM][MAX_KEY_SIZE] = {{""}};
int status[MAX_KEY_SIZE] = {0};

pthread_mutex_t databaseLock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t queuelock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t cond1Lock = PTHREAD_MUTEX_INITIALIZER;

pthread_cond_t cond1 = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond2 = PTHREAD_COND_INITIALIZER;

struct queueNode *front = NULL, *rear = NULL;

int q_count = 0;
int r_count = 0;
int w_count = 0;
int d_count = 0;
int f_count = 0;
int objects = 0;

void print_keys()
{
    for (int i = 0; i < MAX_KEY_NUM; i++)
    {
        printf("%s, ", keys[i]);
    }
    printf("\n");
}

int find_key(char *key)
{
    for (int i = 0; i < MAX_KEY_NUM; i++)
    {
        if (strcmp(keys[i], key) == 0)
        {
            return i;
        }
    }

    return -1;
}

int write_data(char *key, char *data)
{
    FILE *fp;
    char filename[32];

    pthread_mutex_lock(&databaseLock);
    w_count++;
    int seq = find_key(key);
    if (seq == -1)
    {
        for (int i = 0; i < MAX_KEY_NUM; i++)
        {
            if (strcmp(keys[i], "") == 0 && status[i] == 0 && i != 60)
            {
                seq = i;
                strcpy(keys[seq], key);
                objects++;
                break;
            }
        }
    }

    // If busy, return -1.
    printf("seq is %d\n", seq);
    if (status[seq] == 1)
    {
        printf("%s is busy\n", key);
        f_count++;
        pthread_mutex_unlock(&databaseLock);
        return -1;
    }

    // Set the entry to busy.
    status[seq] = 1;
    pthread_mutex_unlock(&databaseLock);

    // Write the data.
    sprintf(filename, "/tmp/data.%d", seq);
    fp = fopen(filename, "w");
    fputs(key, fp);
    fputc('\n', fp);
    fputs(data, fp);
    fputc('\n', fp);
    fclose(fp);

    // Set the entry to not busy.
    pthread_mutex_lock(&databaseLock);
    status[seq] = 0;
    pthread_mutex_unlock(&databaseLock);

    return 0;
}

int read_data(char *key, char **data)
{
    pthread_mutex_lock(&databaseLock);
    r_count++;
    int seq = find_key(key);
    // If the key doesn't exist or is busy, return -1.
    if (seq == -1 || status[seq] == 1)
    {
        if (seq == -1)
        {
            printf("%s doesn't exist\n", key);
        }
        else
        {
            printf("%s is busy\n", key);
        }
        f_count++;
        pthread_mutex_unlock(&databaseLock);
        return -1;
    }

    status[seq] = 1;
    pthread_mutex_unlock(&databaseLock);

    char filename[32];
    FILE *fp;
    size_t len = 0;

    sprintf(filename, "/tmp/data.%d", seq);
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        f_count++;
        return -1;
    }

    // First line is the key.
    // Second line is the data.
    getline(data, &len, fp);
    getline(data, &len, fp);
    (*data)[strlen(*data) - 1] = 0;
    fclose(fp);

    pthread_mutex_lock(&databaseLock);
    status[seq] = 0;
    pthread_mutex_unlock(&databaseLock);

    return 0;
}

int delete_data(char *key)
{
    pthread_mutex_lock(&databaseLock);
    d_count++;
    int seq = find_key(key);
    if (seq == -1 || status[seq] == 1)
    {
        if (seq == -1)
        {
            printf("%s doesn't exist\n", key);
        }
        else
        {
            printf("%s is busy\n", key);
        }
        f_count++;
        pthread_mutex_unlock(&databaseLock);
        return -1;
    }

    status[seq] = 1;

    // Make the key invalid(an empty string) immediately.
    pthread_mutex_unlock(&databaseLock);

    char filename[32];
    sprintf(filename, "/tmp/data.%d", seq);
    int ret = remove(filename);
    if (ret != 0)
    {
        f_count++;
        return -1;
    }

    pthread_mutex_lock(&databaseLock);
    objects--;
    status[seq] = 0;
    strcpy(keys[seq], "");
    pthread_mutex_unlock(&databaseLock);

    return 0;
}

void queue_work(int sock_fd)
{
    struct queueNode *newNode;
    newNode = (struct queueNode *)malloc(sizeof(struct queueNode));
    newNode->sock = sock_fd;
    newNode->next = NULL;

    pthread_mutex_lock(&queuelock);
    if (front == NULL)
    {
        front = rear = newNode;
    }
    else
    {
        rear->next = newNode;
        rear = newNode;
    }
    q_count++;
    pthread_mutex_unlock(&queuelock);

    pthread_cond_signal(&cond1);
    return;
}

int get_work()
{
    if (front == NULL)
    {
        return -1;
    }
    struct queueNode *temp = front;
    int return_sock = temp->sock;
    if (front == rear)
    {
        front = NULL;
        rear = NULL;
    }
    else
    {
        front = front->next;
    }
    free(temp);
    q_count--;
    return return_sock;
}

void handle_work(int sock_fd)
{
    struct request client_request;
    struct request server_response;

    char *read_buffer = NULL;

    if (sock_fd < 0)
    {
        // If the sock_fd is negative throw an error back to the client
        server_response.op_status = 'X';
        f_count++;
        write(sock_fd, &server_response, sizeof(struct request));
        return;
    }

    read(sock_fd, &client_request, sizeof(struct request));
    server_response.op_status = 'K';
    sprintf(server_response.len, "%d", 0);
    strcpy(server_response.name, client_request.name);
    char *key = client_request.name;
    int buffer_len = atoi(client_request.len);
    char dataBuffer[buffer_len];

    switch (client_request.op_status)
    {
    case 'W':
        read(sock_fd, dataBuffer, buffer_len);
        dataBuffer[buffer_len] = 0;
        printf("Writing %s to %s with len %d\n", dataBuffer, client_request.name, buffer_len);
        if (write_data(key, dataBuffer) == -1)
        {
            server_response.op_status = 'X';
        }
        printf("Writing finished\n");
        write(sock_fd, &server_response, sizeof(struct request));
        break;

    case 'R':
        // Read method
        printf("Reading data\n");
        if (read_data(key, &read_buffer) == -1)
        {
            printf("Read for %s failed'\n", key);
            server_response.op_status = 'X';
        }
        else
        {
            sprintf(server_response.len, "%lu", strlen(read_buffer));
            printf("%s, %lu\n", read_buffer, strlen(read_buffer));
        }
        printf("Read finished\n");
        write(sock_fd, &server_response, sizeof(struct request));
        if (read_buffer != NULL)
        {
            write(sock_fd, read_buffer, strlen(read_buffer));
        }
        break;

    case 'D':
        // Delete method
        printf("Deleting %s\n", key);
        if (delete_data(client_request.name) == -1)
        {
            server_response.op_status = 'X';
        }
        printf("delete finished\n");
        write(sock_fd, &server_response, sizeof(struct request));
        break;

    case 'Q':
        // quit method
        close(sock_fd);
        exit(0);
        break;

    default:
        write(sock_fd, &server_response, sizeof(struct request));
        break;
    }

    close(sock_fd);
    usleep(random() % 10000);
}

void *handler(void *args)
{
    while (1)
    {
        pthread_mutex_lock(&queuelock);
        while (q_count == 0)
        {
            pthread_cond_wait(&cond1, &queuelock);
        }
        int sock_fd = get_work();
        pthread_mutex_unlock(&queuelock);

        if (sock_fd != -1)
        {
            handle_work(sock_fd);
        }
        usleep(random() % 10000);
    }
}

void *listener(void *args)
{
    int port = 5000;
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in addr = {.sin_family = AF_INET,
                               .sin_port = htons(port),
                               .sin_addr.s_addr = 0};

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        perror("can't bind"), exit(1);
    }
    if (listen(sock, 2) < 0)
    {
        perror("listen"), exit(1);
    }

    while (1)
    {
        int sock_fd = accept(sock, NULL, NULL);
        queue_work(sock_fd);
        usleep(random() % 10000);
    }
}

int main(int argc, char *argv[])
{
    system("rm -f /tmp/data.*"); // removes all the files and starts a clean db

    pthread_t listener_thread;
    pthread_create(&listener_thread, NULL, listener, NULL);

    pthread_t handler_thread1, handler_thread2, handler_thread3, handler_thread4;
    pthread_create(&handler_thread1, NULL, handler, NULL);
    pthread_create(&handler_thread2, NULL, handler, NULL);
    pthread_create(&handler_thread3, NULL, handler, NULL);
    pthread_create(&handler_thread4, NULL, handler, NULL);

    // This is the main thread where take the commands
    char line[128];
    while (fgets(line, sizeof(line), stdin) != NULL)
    {
        if (strcmp(line, "quit\n") == 0)
        {
            exit(0);
        }
        else if (strcmp(line, "stats\n") == 0)
        {
            printf("Number of objects: \t %d\nNumber of reads: \t %d\nNumber of writes: \t %d\nNumber of deletes: \t %d\nNumber of queued: \t %d\nNumber of fails: \t %d\n", objects, r_count, w_count, d_count, q_count, f_count);
        }
    }

    return 0;
}
