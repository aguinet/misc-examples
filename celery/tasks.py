from celery import Celery

app = Celery('tasks', backend='redis://localhost/', broker='amqp://localhost//')

@app.task()
def task_sum(numbers):
    return sum(numbers)

@app.task()
def task_mul(x, y):
    return x*y
