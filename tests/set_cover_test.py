from cls_python.setcover import minimal_covers, minimal_elements

# TODO: testing framework
contains = lambda s, e: e in s
sets: list[list[int]] = [[1,4],[7,3],[7,9],[0,1,2],[1,3],[2,3],[1,6]]
elements: list[int] = [0,1,2,3,7]
print(minimal_covers(sets, elements, contains))

compare = lambda n1, n2: n2 % n1 == 0  
print(minimal_elements([2,4,22,3,2,3,65,6,7,2,8,9,10,11,12,11,22,2], compare))
