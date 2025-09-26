def select_even(l):
    result = [num for num in l if num%2==0]
    return result

l = [1,2,3,8,19,20]
r = select_even(l)
print(r)

# 考推导式，其实不难，推导式的语法挺符合人常规思维！