<div class="container">
  <div class="row">
    <div class="col-sm">
      <img align="left" src="logo.png" height=90/>
    </div>
    <div class="col">
      <h1 align="justify">
        Semi-supervised labeled sequence transduction with differentiable discrete variables
      </h1>
    </div>
  </div>
</div>

<p align="justify">
A word’s form reflects syntactic and semantic categories that are expressed by the word through a process termed morphology. For example, each English count noun has both singular and plural forms. These are known as the inflected forms. Some languages display little inflection, while others possess a proliferation of forms.
</p>
<p align="justify">
<b>Morphological Reinflection problem</b>: Using a source sequence, x^s which describes the inflected word, and target labels, y^t (morphosyntactic description), we hope to realize the target sequence, x^t which is the re-inflected form of the input sequence.
</p>

<p align="justify">
(Zhou and Neubig, 2017) introduce a new framework for labeled sequence transduction problems: multi-space variational encoder-decoders (<a href="https://github.com/akashrajkn/MSVED-morph-reinflection">MSVED</a>). To explain the observed data, this framework employs continuous and discrete latent variables belonging to multiple separate probability distributions.  For the task of morphological re-inflection, source and target word forms (and lemma) are represented by continuous random variables and each of the morphosyntactic descriptions isrepresented by a discrete random variable.  A challenge in this model is performing backprop through discrete random variables, which is addressed by using the Gumbel-Softmax trick (Maddison et al.,2014; Gumbel and Lieblein, 1954).
</p>

<h3> References </h3>

<ul>
  <li align="justify"> Gumbel, E. J. and Lieblein, J. (1954).  Some applications of extreme-value methods. <i>The American Statistician</i>, 8(5):14–17. </li>

  <li align="justify"> Kibrik, A. E. (1998).  The handbook of morphology. pages 455–476. </li>

  <li align="justify"> Maddison, C. J., Mnih, A., and Teh, Y. W. (2016). The  concrete  distribution:  A  continuous relaxation of discrete random variables. <i>arXiv preprint arXiv:1611.00712</i>. </li>

  <li align="justify"> Maddison, C. J., Tarlow, D., and Minka, T. (2014). A* sampling. In <i>Advances in Neural Information Processing Systems</i>, pages 3086–3094. </li>
  
  <li align="justify"> Zhou, C. and Neubig, G. (2017).  Multi-space variational encoder-decoders for semi-supervised labeled sequence transduction. <i>arXiv preprint arXiv:1704.01691</i>. </li>
</ul>
