from tasks import task_mul, task_sum
import celery

# Compute 4*4
res = task_mul.delay(4,4).get()
print(res)

# Compute 4+4
res = task_sum.delay([4,4]).get()
print(res)

# Compute 4*4 + 8*8
res = celery.chord([task_mul.subtask(args=(4,4)), task_mul.subtask(args=(8,8))])(task_sum.subtask()).get()
print(res)
