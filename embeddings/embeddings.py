database = {}
with open("glove.6B.50d.txt", "r") as file:
    for line in file:
        pieces = line.split()
        key = pieces[0]
        vector = [float(num) for num in pieces[1:]]
        database[key] = vector

# print(len(database))        # how many words did we load?
# print(database["king"][:5]) # first 5 numbers of king's vector

user_input_one = input("Enter the first word: ")
user_input_two = input("Enter the second word you want to subtract from the first: ")
user_input_three = input("Enter the third word you want to add back from the 2 words subtraction: ")

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

# print(cosine_similarity(database["cat"], database["dog"]))

def words_similarity(a, b, c):
    new_word_vector = []
    closest_word = ""
    best_score = -1
    for word_1, word_2, word_3 in zip(database[a], database[b], database[c]):
        new_word_vector.append(word_1 - word_2 + word_3)       
    for word in database:
        if word in [a, b, c]:
            continue
        vector = database[word]
        vector_score = cosine_similarity(vector, new_word_vector)
        if vector_score > best_score:
            closest_word = word
            best_score = vector_score
    return closest_word


# result = words_similarity("king", "man", "woman")
# print(len(result))
# print(result)
# print(words_similarity("king", "man", "woman"))

print(words_similarity(user_input_one, user_input_two, user_input_three))