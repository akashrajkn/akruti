# akruti

A word’s form reflects syntactic and semantic categories \todo{labeled sequence transduction} that are expressed by the word through a process termed morphology. For example, each English count noun has both singular and plural forms. These are known as the inflected forms. Some languages display little inflection, while others possess a proliferation of forms.

**Morphological Reinflection problem**: Using a source sequence, x^s which describes the inflected word, and target labels, y^t (morphosyntactic description), we hope to realize the target sequence, x^t which is the re-inflected form of the input sequence.

(Zhou and Neubig, 2017) introduce a new framework for labeled sequence transduction problems:multi-space variational encoder-decoders (MSVED). To explain the observed data, this frameworkemploys continuous and discrete latent variables belonging to multiple separate probability distri-butions.  For the task of morphological re-inflection, source and target word forms (and lemma)are represented by continuous random variables and each of the morphosyntactic descriptions isrepresented by a discrete random variable.  A challenge in this model is performing backprop throughdiscrete random variables, which is addressed by using the Gumbel-Softmax trick (Maddison et al.,2014; Gumbel and Lieblein, 1954).

(Louizos et al., 2017) proposehard concretedistribution, which is obtained by stretching a binaryconcrete distribution (Maddison et al., 2016) and then transforming its samples with a hard-sigmoid.This distribution is differentiable with respect to its parameters, which allows for straightforwardapplication of gradient descent. 
