database = {}
with open("glove.6B.50d.txt", "r") as file:
    for line in file:
        pieces = line.split()
        key = pieces[0]
        vector = [float(num) for num in pieces[1:]]
        database[key] = vector

print(len(database))        # how many words did we load?
print(database["king"][:5]) # first 5 numbers of king's vector

