from cls_python.setcover import minimal_covers

# TODO: testing framework
contains = lambda s, e: e in s
sets: list[list[int]] = [[1,4],[7,3],[7,9],[0,1,2],[1,3],[2,3],[1,6]]
elements: list[int] = [0,1,2,3,7]
print(minimal_covers(sets, elements, contains))

