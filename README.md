# ml from scratch

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

## Neural Layer

Without knowing calculus and with just the linear algebra I understand, I've managed to write the actual process of each neural layer of an ML model.

**What I built:** A Python program that simulates a single neural network layer (forward pass) and a ReLU activation function.

What `layer.py` does is take the weights and the bias — which for now will stay a "black box" that we get from other steps (the actual training) — and do math and matrix operations on the input vector to get the desired output. What happens in the neural layer function is we take each row of the weights and get its dot product with the input vector. After we get the dot product for each row, we add the corresponding bias to the result.

Now since a neural network is composed of not just one layer but an increasing amount (depends on the task, hardware, etc.), doing the exact same operations won't lead us anywhere since 100 matrix multiplications are mathematically equal to just 1 matrix multiplication. That's where the ReLU function comes in — it zeros out every negative number in the output, which introduces non-linearity. That way, stacking layers actually becomes meaningful because each layer can learn different patterns. An actual neural network at the skeleton level is just a vector passing through `neural_layer()` then `ReLU()` a bunch of times.

**What I learned:** A neural network layer is just a dot product of each weight row with the input, plus a bias — the same linear algebra operations I already built. ReLU adds non-linearity so that stacking layers matters. Without it, any number of layers collapses into one. The weights and biases aren't hand-picked — they're learned during training by feeding the network labeled data (inputs + correct answers) and adjusting the weights thousands of times until the output is right.

## Autograd

Now that I understand the calculus chain rule, I'm building a tiny autograd engine — the thing that lets a neural network actually "learn" by computing gradients automatically.

**What I built:** `Value` is the skeleton I needed in order to implement training. The idea is to take every operator that gets used during training (`+`, `-`, `*`, `**`, etc.) and override its behavior for `Value` objects — a Python pattern called *operator overloading*. This lets a `Value` do more than a regular number: when you compose arithmetic, each result remembers its parents and which operation created it, building a tree-shaped graph as a side effect of the forward pass.

That graph is what makes training possible. Each node carries a small function that knows how to do one step of the chain rule — take the gradient sitting on this node and push it to its parents. After the forward pass builds the graph, we walk it backwards from output to inputs, calling each node's chain-rule step in turn. The result is full gradient propagation across an arbitrary expression, even one with many steps.

**What I learned:** Connecting this back to ML, I learned that to understand the actual change from the input to the output, we have to calculate the gradient of each function using the chain rule. But the chain rule alone isn't enough — since it's a multi-step process, we have to "save" each step along the way. This is where saving each parent and the operation used comes in handy. It lets us build a tree-shaped graph as we go forward, and that saved graph is what enables us to walk backwards through it. This is where the term *backpropagation* comes from.

When the machine walks backwards, it calculates the gradients. A separate step then uses those gradients as a "guideline" to adjust the knobs (`w`, `b`) in the direction that reduces the loss. The full training loop runs forward to get the prediction and the loss, then backwards to compute the gradients, then the descent step adjusts the knobs (bigger loss → bigger adjustment). Looped enough times, the loss approaches 0 and the model has "learned."

## Single Neuron Training

**What I built:** After creating the `Value` class, I can use it to train a "model" — a single-neuron one, but still a model. What `train.py` does is take a list of inputs paired with targets, then wraps the knobs (`w`, `b`) as `Value` objects so they live in the autograd graph. The knobs start random each run.

Setting the learning rate is a balancing act: too big and the updates overshoot the answer and oscillate; too small and each step barely moves, so it takes far more iterations to converge.

After setting parameters, the loop runs:
- **Forward pass** — accumulate loss across all training inputs (MSE)
- **Zero out gradients** — autograd uses `+=` for accumulation, so without this every loop would compound the gradients from previous loops
- **Backward pass** — propagate gradients back through the graph
- **Update knobs** — `w` and `b` step in the opposite direction of their gradients, scaled by the learning rate

The next loop runs with the updated knobs and tries again. After enough iterations, `w` and `b` settle on values that fit all the training points.

Once training is done, the model gets presented with inputs that *weren't* in the training data. It still makes good predictions — and that's what makes this an actual model and not a lookup table that can only answer about its training set.

**What I learned:** This phase was less learning-intensive than the ones before it — it was mostly implementing on top of foundations already in place. But running the generalization test and seeing it work in real time reinforced what makes ML different from a lookup table: the model doesn't memorize training points, it finds a function that fits all of them, and that function then applies to new inputs it has never seen.

I also picked up the learning rate trade-off first-hand — too small and training drags, too large and the model overshoots and the loss explodes. Theorizing about why a fixed learning rate is fragile led me to discover that smarter optimizers exist: ones that adjust the learning rate dynamically, or use the recent gradient history to dampen oscillations, or both. Plain gradient descent (what I implemented) is the simplest possible thing — and the reason fancier optimizers like Adam exist is exactly the fragility I ran into. 