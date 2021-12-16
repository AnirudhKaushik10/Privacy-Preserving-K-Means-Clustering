Privacy preserving K-means clustering system for 2 party computation in Python by utilizing Pytorch and Facebook’s secure computation library Crypten.

The use case of a privacy preserving K Means clustering model arises when multiple parties want to train a joint model by combining their individual datasets but none of them want to compromise the privacy of their datasets. In order to train such a model, we utilize restructured mathematical operations using secret sharing to make them secure with the help of the Oblivious Transfer protocol, which guarantees the preservation of the privacy of individual datasets.


Install the Dependencies =>

Clone https://github.com/facebookresearch/CrypTen
pip3 install .
pip install matplotlib

How to Run =>

python launcher.py
Runs the Encrypted K Means model on mall customer dataset. According to the experiments, the optimal k value = 5

python main.py

Runs the normal K Means model on mall customer dataset.
Results between the two datasets can be compared at the end of each run.



Secret Sharing using Oblivious Transfer Protocol which is used to train the model =>

• Addition
We have used the additive sharing scheme to hide the values held by 2 parties. Assuming two parties
Alice and Bob have values x and y, each party gets shares [x] and [y], say [x1] [y1] and [x2] [y2], such
that it is only possible to reveal the values if and only if both parties agree to reveal their shares. In
such a scheme, secure addition can be performed locally such that Alice computes [x1] + [y1] and Bob
computes [x2] + [y2] and they add these shares together to find the result of x + y.

• Subtraction
Subtraction is computed in a similar way to addition under additive secret sharing scheme. Alice and
Bob can locally compute the shares of [x1] - [y1] and [x2] - [y2] and then add them together to get the
value of x - y.

• Multiplication
Using Crypten, secure multiplication of two values held by two parties, say Alice and Bob is done with
the help of beaver’s triples.
First a triplet (a,b,c) is generated in such a way that c = a * b
Let’s assume, Alice has the value x and Bob has the value y, and they wish to compute z = x * y,
without revealing each other’s values. Furthermore, let’s have u = x + y and v = x - y (this can be
locally computed by Alice and Bob, assuming that x and y have been additive shared among them.)
Then each party has a share of u and v, denoted by [u] and [v] respectively.
Each party is, therefore, able to compute locally [d] = [u] - [a] and [e] = [v] - [b], where [a] and [b] are
the shares of beaver’s triples.
Values of d and e are reconstructed, thereby making them publicly available.
Again, the parties locally calculate [u*v] = d * [a] + e * [b] + [c] +d * e, and thus they have shares of
the multiplication.

• Square
Square is calculated in a similar way to multiplication. In this case random sharing of [r] and [r2] are
generated, in such a way that [r2] = [r * r]. The value [x](whose square is to be calculated) is additively
hidden via [r] and publicly known [e] is defined such that [e] = [x] - [r]
Then the result of the square of (x), say z, is calculated as
[z] = [r2] + 2 * e * [r] + e * e.

• Division
In k means clustering, division operation is used during the calculation of the centroid of clusters, in
this case each party already knows the number of elements present in the cluster, hence we only require
to divide the distance by the publicly known cluster size.
