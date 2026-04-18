# ml-from-scratch

A side project I'm working on to understand AI, ML, and deep learning from the ground up. Zero expectations — just genuine curiosity. The long-term goal is to eventually build my own small ML program from scratch.

## Linear Algebra

The first waystone on the path. Contains small Python programs that help me visualize the math rather than just doing calculations by hand. Also a good excuse to get more comfortable with Python.

**What I built:** Matrix multiplication from scratch, and a visualization of linear transformations using matplotlib — showing how a transformation matrix warps a grid.

**What I learned:** How to apply matrix multiplication in code, why columns represent basis vectors, and what a transformation actually looks like geometrically.

## Embeddings

After understanding some core linear algebra I went ahead and tried my hand at embeddings, which utilizes taking words and assigning a vector to them. By assigning a vector to them we can then do arithmetic functions on them like normal numbers. So by that logic we can take a word, subtract another word's value from it, add a 3rd word, and get as a result a 4th word with a similar meaning.

**What I built:** A Python program that takes in 3 words and returns a 4th word that corresponds to the math being done.

First, to be able to do any sort of math we have to convert words to vectors — this is already done by the database imported. After getting the 2 vectors we need to find their dot product. This number can tell us how close or far the vectors are from each other (how related the words are). This is a number we save for now. The main problem we get from having just the dot product is that it's influenced by the length of a vector, so we have to strip the length from the vectors and make them all the same length, thus not skewing results by length — which is determined by the training data giving some words that appear more often a larger vector. This would make some words that just appear more often have a higher dot product, which will make us think they are related when in fact they just get a high score from their vector length. So to counteract that we have to get each vector's length — this is its magnitude. We take both vectors' magnitudes (their lengths) and multiply them by each other. Then we divide the dot product by the multiplied magnitudes. This is the cosine similarity of those vectors. It will give us a number between -1 and 1. -1 being the vectors are totally pointing in opposite directions which means the words are exactly opposite in meaning, 1 meaning the words have the exact same meaning, and 0 meaning the words are unrelated.

**What I learned:** I learned so much building this program. First I dove in to understand how those datasets are built — each word gets its own exact vector. First I had to save the database in a way where each word was a key and the value would be its vector. After that is where the fun starts — I had to apply the concepts I learned about in linear algebra to start transforming the word's vectors to actually get meaningful stuff. Like getting the dot product, which tells us in which direction and how close 2 vectors point. Then to actually try to see if words have similar meaning, I found out some words' vectors are longer than others, which happens from those words appearing more often in the datasets given, so they appear to have more "weight." This is actually bad for my program since 2 words with larger vectors, even if unrelated, would show as related since their vectors are actually closer (apple appears 100 times, pear appears 3 times — apple's vector is larger, thus can appear close to an unrelated word that also appears 100 times). So I had to find a way to get rid of the length of a vector for the similarity check. I find each vector's magnitude, which is just its length. After finding the length of those vectors, we multiply the 2 words' magnitudes by each other, then divide the dot product by the multiplied magnitudes. This result will span between -1 and 1. This number tells me the relation between those 2 words — 1 being very related, 0 being unrelated, and -1 being opposites.

Lastly, by getting all this we can do a cool trick: take 2 words, subtract one of their vectors, then add another word's vector — finding a whole new word in the process which is related to the first word without the 2nd word's context and with the 3rd word's context added on top.

For example:
- king - man + woman = queen
- france - paris + japan = tokyo