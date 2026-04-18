database = {}
with open("glove.6B.50d.txt", "r") as file:
    for line in file:
        pieces = line.split()
        key = pieces[0]
        vector = [float(num) for num in pieces[1:]]
        database[key] = vector

print(len(database))        # how many words did we load?
print(database["king"][:5]) # first 5 numbers of king's vector

def dot_product(a, b):
    dot_product_result = 0
    for x, y in zip(a, b):
        dot_product_result += (x * y)
    return dot_product_result

def magnitude(v):
    return sum(x**2 for x in v) ** 0.5


def cosine_similarity(a, b):
    dot_product_result = dot_product(a, b)
    magnitude_a = magnitude(a)
    magnitude_b = magnitude(b)
    result = dot_product_result / (magnitude_a * magnitude_b)
    return result

print(cosine_similarity(database["cat"], database["dog"]))

def words_similarity(a, b, c):
    result = []
    for word_1, word_2, word_3 in zip(database[a], database[b], database[c]):
        result.append(word_1 - word_2 + word_3)       
    return result

result = words_similarity("king", "man", "woman")
print(len(result))
print(result)
# print(words_similarity("king", "man", "woman"))