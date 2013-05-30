import math

def pos(n):
    i = int(math.sqrt(2*n))
    if (i * (i+1) / 2 < n):
        i = i + 1
    line = i - 1
    col = n - line * (line + 1) / 2 - 1
    return [line, col]

for i in range(1,17):
    print(pos(i))

print(pos(28))
