# Parts of Speech Tagging

Part 1: Part-of-speech tagging
Natural language processing (NLP) is an important research area in artificial intelligence, dating
back to at least the 1950’s. One of the most basic problems in NLP is part-of-speech tagging, in
which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective,
etc.). This is a first step towards extracting semantics from natural language text. For example,
consider the following sentence:
Her position covers a number of daily tasks common to any social director.
Part-of-speech tagging here is not easy because many of these words can take on different parts of
speech depending on context. For example, position can be a noun (as in the above sentence) or
a verb (as in “They position themselves near the exit”). In fact, covers, number, and tasks can all
be used as either nouns or verbs, while social and common can be nouns or adjectives, and daily
can be an adjective, noun, or adverb. The correct labeling for the above sentence is:
Her position covers a number of daily tasks common to any social director.
DET NOUN VERB DET NOUN ADP ADJ NOUN ADJ ADP DET ADJ NOUN
where DET stands for a determiner, ADP is an adposition, ADJ is an adjective, and ADV is an
adverb.1 Labeling parts of speech thus involves an understanding of the intended meaning of the
words in the sentence, as well as the relationships between the words.
Fortunately, some relatively simple statistical models can do amazingly well at solving NLP problems. In particular, consider the Bayesian network shown in Figure 1(a). This Bayes net has a set
of N random variables S = {S1, . . . , SN } and N random variables W = {W1, . . . , WN }. The W
variables represent observed words in a sentence, so Wi ∈ {w|w is a word in the English language}.
The variables in S represent part of speech tags, so Si ∈ {VERB, NOUN, . . .}. The arrows between
W and S nodes model the probabilistic relationship between a given observed word and the possible parts of speech it can take on, P(Wi
|Si). (For example, these distributions can model the
fact that the word “dog” is a fairly common noun but a very rare verb. The arrows between S
nodes model the probability that a word of one part of speech follows a word of another part of
speech, P(Si+1|Si). (For example, these arrows can model the fact that verbs are very likely to
follow nouns, but are unlikely to follow adjectives.)
We can use this model to perform part-of-speech tagging as follows. Given a sentence with N
words, we construct a Bayes Net with 2N nodes as above. The values of the variables W are just
the words in the sentence, and then we can perform Bayesian inference to estimate the variables in
S.
1
If you didn’t know the term “adposition”, neither did I. The adpositions in English are prepositions; in many
languages, there are postpositions too. But you won’t need to understand the linguistic theory between these parts
of speech to complete the assignment; if you’re curious, you might check out the “Part of Speech” Wikipedia article
for some background.
2
S1 S2 S3 S4 SN
W1 W2 W3 W4 WN
... S1 S2 S3 S4 SN
W1 W2 W3 W4 WN
... S1 S2 S3 S4 SN
W1 W2 W3 W4 WN
...
(a) (b) (c)
Figure 1: Three Bayes Nets for part of speech tagging: (a) an HMM, (b) a simplified model, and
(c) a more complicated model.
Data. To help you with this assignment, we’ve prepared a large corpus of labeled training and
testing data, consisting of nearly 1 million words and 50,000 sentences. The file format of the
datasets is quite simple: each line consists of a word, followed by a space, followed by one of
12 part-of-speech tags: ADJ (adjective), ADV (adverb), ADP (adposition), CONJ (conjunction),
DET (determiner), NOUN, NUM (number), PRON (pronoun), PRT (particle), VERB, X (foreign
word), and . (punctuation mark). Sentence boundaries are indicated by blank lines.2
What to do. Your goal in this part is to implement part-of-speech tagging in Python, using Bayes
networks.
1. First you’ll need to estimate the probabilities of the HMM above, namely P(S1), P(Si+1|Si),
and P(Wi
|Si). To do this, use the labeled training file we’ve provided.
2. Your goal now is to label new sentences with parts of speech, using the probability distributions learned in step 1. To get started, consider the simplified Bayes net in Figure 1(b).
To perform part-of-speech tagging, we’ll want to estimate the most-probable tag s∗i for each word Wi,
s
∗
i = arg max
si
P(Si = si
|W).
Implement part-of-speech tagging using this simple model. Hint: This is easy; if you don’t
see why, try running Variable Elimination by hand on the Bayes Net in Figure 1(b).
3. Now consider the Bayes net of Figure 1(a), which is a richer model that incorporates dependencies between words. Implement the Viterbi algorithm to find the maximum a posteriori
(MAP) labeling for the sentence – i.e. the most likely state sequence:
(s∗1, . . . , s∗N ) = arg maxs1,...,sNP(Si = si|W).

Consider the Bayes Net of Figure 1c, which is a better model of language because it incorporates some longer-term dependencies between words. It’s no longer an HMM, so one
can’t use Viterbi, but we can use MCMC. Write code that uses MCMC to sample from the

This dataset is based on the Brown corpus. Modern part-of-speech taggers often use a much larger set of tags –
often over 100 tags, depending on the language of interest – that carry finer-grained information like the tense and
mood of verbs, whether nouns are singular or plural, etc. In this assignment we’ve simplified the set of tags to the
12 described here; the simple tag set is due to Petrov, Das and McDonald, and is discussed in detail in their 2012
LREC paper if you’re interested.

Figure 2: In OCR, our goal is to extract text from a noisy scanned image of a document.
posterior distribution of Fig 1c, P(S|W), after a warm-up period, and shows five sampled
particles. Then estimate the best labeling for each word (by picking the maximum marginal
for each word, s
∗
i = arg maxsi P(Si = si
|W), as in step 2). (To do this, just generate many
(thousands?) of samples and, for each individual word, check which part of speech occurred
most often.)
Your program should take as input two filenames: a training file and a testing file. The program
should use the training corpus for Step 1, and then display the output of Steps 2-4 on each sentence
in the testing file. For the result generated by each of the three approaches (Simple, HMM, and
Complex), as well as for the ground truth result, your program should output the logarithm of the
posterior probability for each solution it finds under each of the three models in Figure 1. It should
also display a running evaluation showing the percentage of words and whole sentences that have
been labeled correctly according to the ground truth so far. For example:
[djcran@raichu djc-sol]$ ./label.py training_file testing_file
Learning model...
Loading test data...
Testing classifiers...
Simple HMM Complex Magnus ab integro seclorum nascitur ordo .
0. Ground truth -48.52 -64.33 -78.21 noun verb adv conj noun noun .
1. Simple -47.29 -66.74 -79.01 noun noun noun adv verb noun .
2. HMM -47.48 -63.83 -79.12 noun verb adj conj noun verb .
3. Complex -48.52 -64.33 -78.21 noun verb adv conj noun noun .
==> So far scored 1 sentences with 17 words.
Words correct: Sentences correct:
0. Ground truth: 100.00% 100.00%
1. Simplified: 42.85% 0.00%
2. HMM MAP: 71.43% 0.00%
3. Complex MCMC: 100.00% 100.00%
We’ve already implemented some skeleton code to get you started, in three files: label.py, which is
the main program, pos scorer.py, which has the scoring code, and pos solver.py, which will contain
the actual part-of-speech estimation code. You should only modify the latter of these files; the
current version of pos solver.py we’ve supplied is very simple, as you’ll see. In your report, please
make sure to include your results (accuracies) for each technique on the test file we’ve supplied,
bc.test.
