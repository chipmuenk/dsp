# Fibonacci series:
# the sum of two elements defines the next
a, b = 0, 1
c = d = e = a+1
print c, d, e
while b < 10:
     print a, b
     a, b = b, a+b
print "Ende!"

