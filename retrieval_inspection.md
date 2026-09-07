# NEXUS Retrieval Inspection — Step 4

Read-only inspection using production `get_or_create_store()` + `get_mmr_retriever()`. No LLM invoked.

Collection: `nexus_collection` (241 chunks)

## QUESTION Q2

### Question
What is the exact count of total trainable parameters in the RAG models described by Patrick Lewis et al., and how are these parameters distributed among its sub-components?

### Retrieved Chunk 1

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 17
Chunk ID: 2f35dbe1b36046065203093ad24253c1

```text
logit by learning (i) a document embedding for the null document, (ii) a static learnt bias term, or
(iii) a neural network to predict the logit. We did not ﬁnd that these improved performance, so in
the interests of simplicity, we omit them. For Open MS-MARCO, where useful retrieved documents
cannot always be retrieved, we observe that the model learns to always retrieve a particular set of
documents for questions that are less likely to beneﬁt from retrieval, suggesting that null document
mechanisms may not be necessary for RAG.
G Parameters
Our RAG models contain the trainable parameters for the BERT-base query and document encoder of
DPR, with 110M parameters each (although we do not train the document encoder ourselves) and
406M trainable parameters from BART-large, 406M parameters, making a total of 626M trainable
18
```

### Retrieved Chunk 2

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 18
Chunk ID: af27ea181bce9254e0289fe7a6f43076

```text
The non-parametric memory index does not consist of trainable parameters, but does consists of 21M
728 dimensional vectors, consisting of 15.3B values. These can be easily be stored at 8-bit ﬂoating
point precision to manage memory and disk footprints.
H Retrieval Collapse
In preliminary experiments, we observed that for some tasks such as story generation [ 11], the
retrieval component would “collapse” and learn to retrieve the same documents regardless of the
input. In these cases, once retrieval had collapsed, the generator would learn to ignore the documents,
and the RAG model would perform equivalently to BART. The collapse could be due to a less-explicit
requirement for factual knowledge in some tasks, or the longer target sequences, which could result
in less informative gradients for the retriever. Perez et al.[46] also found spurious retrieval results
when optimizing a retrieval component in order to improve performance on downstream tasks.
I Number of instances per dataset
```

### Retrieved Chunk 3

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 0
Chunk ID: 6bfee6d2163efb0544a28287e121b860

```text
Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks
Patrick Lewis†‡, Ethan Perez⋆,
Aleksandra Piktus†, Fabio Petroni†, Vladimir Karpukhin†, Naman Goyal†, Heinrich Küttler†,
Mike Lewis†, Wen-tau Yih†, Tim Rocktäschel†‡, Sebastian Riedel†‡, Douwe Kiela†
†Facebook AI Research;‡University College London;⋆New York University;
plewis@fb.com
Abstract
Large pre-trained language models have been shown to store factual knowledge
in their parameters, and achieve state-of-the-art results when ﬁne-tuned on down-
stream NLP tasks. However, their ability to access and precisely manipulate knowl-
edge is still limited, and hence on knowledge-intensive tasks, their performance
lags behind task-speciﬁc architectures. Additionally, providing provenance for their
decisions and updating their world knowledge remain open research problems. Pre-
trained models with a differentiable access mechanism to explicit non-parametric
```

### Retrieved Chunk 4

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 18
Chunk ID: e8948b4aa86d123e503ea6ef7075351e

```text
Table 7: Number of instances in the datasets used. *A hidden subset of this data is used for evaluation
Task Train Development Test
Natural Questions 79169 8758 3611
TriviaQA 78786 8838 11314
WebQuestions 3418 362 2033
CuratedTrec 635 134 635
Jeopardy Question Generation 97392 13714 26849
MS-MARCO 153726 12468 101093*
FEVER-3-way 145450 10000 10000
FEVER-2-way 96966 6666 6666
parameters. The best performing "closed-book" (parametric only) open-domain QA model is T5-11B
with 11 Billion trainable parameters. The T5 model with the closest number of parameters to our
models is T5-large (770M parameters), which achieves a score of 28.9 EM on Natural Questions [52],
substantially below the 44.5 that RAG-Sequence achieves, indicating that hybrid parametric/non-
parametric models require far fewer trainable parameters for strong open-domain QA performance.
The non-parametric memory index does not consist of trainable parameters, but does consists of 21M
```

## QUESTION Q5

### Question
What are the mathematical formula and the specific learning rate warmup steps value used to configure the learning rate scheduler in the Transformer model training?

### Retrieved Chunk 1

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 6
Chunk ID: 3fc99d8e0ce51c516bc2395e8e708173

```text
rate over the course of training, according to the formula:
lrate =d−0.5
model· min(step_num−0.5,step _num·warmup_steps−1.5) (3)
This corresponds to increasing the learning rate linearly for the ﬁrstwarmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.
5.4 Regularization
We employ three types of regularization during training:
Residual Dropout We apply dropout [27] to the output of each sub-layer, before it is added to the
sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
Pdrop = 0.1.
7
```

### Retrieved Chunk 2

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 6
Chunk ID: 5ae525d61dca6d8aa8be8f9430999941

```text
2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece
vocabulary [31]. Sentence pairs were batched together by approximate sequence length. Each training
batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000
target tokens.
5.2 Hardware and Schedule
We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using
the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We
trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the
bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps
(3.5 days).
5.3 Optimizer
We used the Adam optimizer [17] withβ1 = 0.9,β2 = 0.98 andϵ = 10−9. We varied the learning
rate over the course of training, according to the formula:
lrate =d−0.5
model· min(step_num−0.5,step _num·warmup_steps−1.5) (3)
```

### Retrieved Chunk 3

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 2
Chunk ID: 5b51b8ddf2d3a322a7325c224f0a89b5

```text
Figure 1: The Transformer - model architecture.
wise fully connected feed-forward network. We employ a residual connection [10] around each of
the two sub-layers, followed by layer normalization [ 1]. That is, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
layers, produce outputs of dimensiondmodel = 512.
Decoder: The decoder is also composed of a stack ofN = 6 identical layers. In addition to the two
sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
```

### Retrieved Chunk 4

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 0
Chunk ID: fc33946e5210a99f0ae5db5625b82e88

```text
efforts have since continued to push the boundaries of recurrent language models and encoder-decoder
architectures [31, 21, 13].
∗Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started
the effort to evaluate this idea. Ashish, with Illia, designed and implemented the ﬁrst Transformer models and
has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head
attention and the parameter-free position representation and became the other person involved in nearly every
detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and
tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and
efﬁcient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and
```

## QUESTION Q8

### Question
Synthesize a comparative analysis of the computational environments, hardware (GPU models and quantity), and training runtimes utilized to train the Transformer (base vs. big) and the RAG model configurations.

### Retrieved Chunk 1

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 16
Chunk ID: ca1816a105de605ee5c364a4e249e01e

```text
training and inference can be run on one GPU. We ﬁnd that doing Maximum Inner Product Search
with FAISS is sufﬁciently fast on CPU, so we store document index vectors on CPU, requiring∼ 100
GB of CPU memory for all of Wikipedia. After submission, We have ported our code to HuggingFace
Transformers [66]3, which achieves equivalent performance to the previous version but is a cleaner
and easier to use implementation. This version is also open-sourced. We also compress the document
index using FAISS’s compression tools, reducing the CPU memory requirement to 36GB. Scripts to
run experiments with RAG can be found athttps://github.com/huggingface/transformers/
blob/master/examples/rag/README.md and an interactive demo of a RAG model can be found
at https://huggingface.co/rag/
2https://github.com/pytorch/fairseq
3https://github.com/huggingface/transformers
17
```

### Retrieved Chunk 2

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 16
Chunk ID: 8a9a2662fcea6fca4fd87e89ef9cce4b

```text
and a worked example appear when clicking "view tool guide".
Figure 4 shows the user interface for human evaluation. To avoid any biases for screen position,
which model corresponded to sentence A and sentence B was randomly selected for each example.
Annotators were encouraged to research the topic using the internet, and were given detailed instruc-
tions and worked examples in a full instructions tab. We included some gold sentences in order to
assess the accuracy of the annotators. Two annotators did not perform well on these examples and
their annotations were removed from the results.
C Training setup Details
We train all RAG models and BART baselines using Fairseq [45].2 We train with mixed precision
ﬂoating point arithmetic [ 40], distributing training across 8, 32GB NVIDIA V100 GPUs, though
training and inference can be run on one GPU. We ﬁnd that doing Maximum Inner Product Search
```

### Retrieved Chunk 3

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 6
Chunk ID: 5ae525d61dca6d8aa8be8f9430999941

```text
2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece
vocabulary [31]. Sentence pairs were batched together by approximate sequence length. Each training
batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000
target tokens.
5.2 Hardware and Schedule
We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using
the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We
trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the
bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps
(3.5 days).
5.3 Optimizer
We used the Adam optimizer [17] withβ1 = 0.9,β2 = 0.98 andϵ = 10−9. We varied the learning
rate over the course of training, according to the formula:
lrate =d−0.5
model· min(step_num−0.5,step _num·warmup_steps−1.5) (3)
```

### Retrieved Chunk 4

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 9
Chunk ID: 297a737f9f669fde10d9158da0ecb97a

```text
to a lesser extent, including that it might be used to generate abuse, faked or misleading content in
the news or on social media; to impersonate others; or to automate the production of spam/phishing
content [54]. Advanced language models may also lead to the automation of various jobs in the
coming decades [16]. In order to mitigate these risks, AI systems could be employed to ﬁght against
misleading content and automated spam/phishing.
Acknowledgments
The authors would like to thank the reviewers for their thoughtful and constructive feedback on this
paper, as well as HuggingFace for their help in open-sourcing code to run RAG models. The authors
would also like to thank Kyunghyun Cho and Sewon Min for productive discussions and advice. EP
thanks supports from the NSF Graduate Research Fellowship. PL is supported by the FAIR PhD
program.
References
[1] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan
```

## QUESTION Q9

### Question
How do the high-level policy governance requirements for third-party supply chain risk (C-SCRM) in the NIST CSF 2.0 compare to the specific technical vulnerabilities addressed by OWASP Top Ten 2021 Category A08 (Software & Data Integrity Failures)?

### Retrieved Chunk 1

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 11
Chunk ID: 82a594af525a7c323beafd2502418796

```text
A08: Software & Data Integrity Failures
“Software and data integrity failures relate to code and infrastructure that does not 
protect against integrity violations. An example of this is where an application 
relies upon plugins, libraries, or modules from untrusted sources, repositories, and 
content delivery networks (CDNs).” *from the OWASP Top Ten page
Sounds like Software Supply Chain but really isn’t
Difference is that these vulnerabilities leverage execution within. 
This groups vulnerabilities injected into the payloads of otherwise benign artifacts 
before endpoint delivery  - for example
Insecure Deserialization - the log4j vulnerability this week
```

### Retrieved Chunk 2

Source: NIST.CSWP.29.pdf
Page: 17
Chunk ID: 4de0f5c2f9748583a635167c20a4a689

```text
manufacture, acquire, deliver, integrate, operate, maintain, dispose of, and otherwise 
utilize or manage technology products and services. These interactions are shaped and 
influenced by technologies, laws, policies, procedures, and practices. 
Given the complex and interconnected relationships in this ecosystem, supply chain risk 
management (SCRM) is critical for organizations. Cybersecurity SCRM (C-SCRM) is a 
systematic process for managing exposure to cybersecurity risk throughout supply 
chains and developing appropriate response strategies, policies, processes, and 
procedures. The Subcategories within the CSF C-SCRM Category [GV.SC] provide a 
connection between outcomes that focus purely on cybersecurity and those that focus
```

### Retrieved Chunk 3

Source: NIST.CSWP.29.pdf
Page: 22
Chunk ID: c6e85e76db01e3f76e900b81a84a2760

```text
NIST CSWP 29  The NIST Cybersecurity Framework (CSF) 2.0 
February 26, 2024 
 
  18 
o GV.SC-05: Requirements to address cybersecurity risks in supply chains are established, 
prioritized, and integrated into contracts and other types of agreements with suppliers 
and other relevant third parties 
o GV.SC-06: Planning and due diligence are performed to reduce risks before entering into 
formal supplier or other third-party relationships 
o GV.SC-07: The risks posed by a supplier, their products and services, and other third 
parties are understood, recorded, prioritized, assessed, responded to, and monitored 
over the course of the relationship 
o GV.SC-08: Relevant suppliers and other third parties are included in incident planning, 
response, and recovery activities 
o GV.SC-09: Supply chain security practices are integrated into cybersecurity and 
enterprise risk management programs, and their performance is monitored throughout 
the technology product and service life cycle
```

### Retrieved Chunk 4

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 14
Chunk ID: 010d15e4dc20eb266ea72418282ddfd2

```text
Use Case OWASP Top 10 2021 OWASP Application Security Verification Standard
Awareness Yes 
Training Entry level Comprehensive
Design and architecture Occasionally Yes
Coding standard Bare minimum Yes
Secure Code review Bare minimum Yes
Peer review checklist Bare minimum Yes
Unit testing Occasionally Yes
Integration testing Occasionally Yes
Penetration testing Bare minimum Yes
Tool support Bare minimum Yes
Secure Supply Chain Occasionally Yes
```

## QUESTION Q10

### Question
Based on RAG's temporal evaluation experiments, how did swapping Wikipedia index versions (December 2016 vs. December 2018) impact accuracy when querying about world leaders whose positions had changed between those dates?

### Retrieved Chunk 1

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 7
Chunk ID: 48fe46c13a9ea2a9424aa93d41bd5b49

```text
RAG-Token 43.5 54.8 46.5 51.9 17.9 22.6 56.2 49.4 74.5 90.6RAG-Sequence 44.0 55.8 44.9 53.4 15.3 21.5 57.2 47.5
between these dates and use a template “Who is {position}?” (e.g. “Who is the President of Peru?”)
to query our NQ RAG model with each index. RAG answers 70% correctly using the 2016 index for
2016 world leaders and 68% using the 2018 index for 2018 world leaders. Accuracy with mismatched
indices is low (12% with the 2018 index and 2016 leaders, 4% with the 2016 index and 2018 leaders).
This shows we can update RAG’s world knowledge by simply replacing its non-parametric memory.
Effect of Retrieving more documents Models are trained with either 5 or 10 retrieved latent
documents, and we do not observe signiﬁcant differences in performance between them. We have the
ﬂexibility to adjust the number of retrieved documents at test time, which can affect performance and
runtime. Figure 3 (left) shows that retrieving more documents at test time monotonically improves
```

### Retrieved Chunk 2

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 6
Chunk ID: 1bd602768374dedad995fa163db187ac

```text
p(z|x). Table 6 shows the results. For FEVER, BM25 performs best, perhaps since FEVER claims are
heavily entity-centric and thus well-suited for word overlap-based retrieval. Differentiable retrieval
improves results on all other tasks, especially for Open-Domain QA, where it is crucial.
Index hot-swapping An advantage of non-parametric memory models like RAG is that knowledge
can be easily updated at test time. Parametric-only models like T5 or BART need further training to
update their behavior as the world changes. To demonstrate, we build an index using the DrQA [5]
Wikipedia dump from December 2016 and compare outputs from RAG using this index to the newer
index from our main results (December 2018). We prepare a list of 82 world leaders who had changed
7
```

### Retrieved Chunk 3

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 9
Chunk ID: b371e0386acb93bc5880af740dd48b19

```text
Broader Impact
This work offers several positive societal beneﬁts over previous work: the fact that it is more
strongly grounded in real factual knowledge (in this case Wikipedia) makes it “hallucinate” less
with generations that are more factual, and offers more control and interpretability. RAG could be
employed in a wide variety of scenarios with direct beneﬁt to society, for example by endowing it
with a medical index and asking it open-domain questions on that topic, or by helping people be more
effective at their jobs.
With these advantages also come potential downsides: Wikipedia, or any potential external knowledge
source, will probably never be entirely factual and completely devoid of bias. Since RAG can be
employed as a language model, similar concerns as for GPT-2 [50] are valid here, although arguably
to a lesser extent, including that it might be used to generate abuse, faked or misleading content in
```

### Retrieved Chunk 4

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 1
Chunk ID: 34f515fef9217547c28a6861d56490d6

```text
ers Library [66] and can be found at https://github.com/huggingface/transformers/blob/master/
examples/rag/. An interactive demo of RAG models can be found at https://huggingface.co/rag/
2
```

## QUESTION Q11

### Question
Detail the transitions and merges of OWASP Top Ten security categories from the 2017 edition to the 2021 edition. Specifically, identify which three 2017 categories were subsumed into other classifications.

### Retrieved Chunk 1

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 8
Chunk ID: 0448ab746496604f4a67a1e65a8c7a4f

```text
OWASP Top 10 2017 to 2021
● Rank updates
● New categories
● Expanded categories
● Focuses on root causes when possible
```

### Retrieved Chunk 2

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 0
Chunk ID: 7942addf84cad4160fcb8dd18ceb1e70

```text
OWASP Top Ten 2021
Where we’ve been and where we are
```

### Retrieved Chunk 3

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 13
Chunk ID: 905414288f309e85b588ef4c78fe6e41

```text
Subsumed 2017 categories
Cross-Site Scripting – now in Injection
Insecure Deserialization – Software and Data Integrity Failures
XML External Entities – now in Security Misconfiguration
```

### Retrieved Chunk 4

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 15
Chunk ID: 7b41335da05336246bdb449fe116fb2a

```text
How to use the list?
● Evolution of the list has changed. How should we use it?
○ Less specific over time
○ OWASP Top Ten Vulnerabilities doesn’t mean the same any more
■ SAST/DAST Filters and how many results does this mean
■ Standards or Frameworks that say no OWASP Top Ten
```

## QUESTION Q12

### Question
Summarize the English-to-German and English-to-French WMT 2014 translation test results (BLEU scores) for both the Transformer base and big configurations, and identify which previous state-of-the-art models they outperformed.

### Retrieved Chunk 1

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 7
Chunk ID: 7bd7a6e5c646304c98c95a4896010f1d

```text
6 Results
6.1 Machine Translation
On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big)
in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0
BLEU, establishing a new state-of-the-art BLEU score of 28.4. The conﬁguration of this model is
listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model
surpasses all previously published models and ensembles, at a fraction of the training cost of any of
the competitive models.
On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0,
outperforming all of the previously published single models, at less than 1/4 the training cost of the
previous state-of-the-art model. The Transformer (big) model trained for English-to-French used
dropout ratePdrop = 0.1, instead of 0.3.
For the base models, we used a single model obtained by averaging the last 5 checkpoints, which
```

### Retrieved Chunk 2

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 7
Chunk ID: 61d9274b2b272edb07d6d8f8630897d0

```text
Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the
English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.
Model
BLEU Training Cost (FLOPs)
EN-DE EN-FR EN-DE EN-FR
ByteNet [15] 23.75
Deep-Att + PosUnk [32] 39.2 1.0· 1020
GNMT + RL [31] 24.6 39.92 2.3· 1019 1.4· 1020
ConvS2S [8] 25.16 40.46 9.6· 1018 1.5· 1020
MoE [26] 26.03 40.56 2.0· 1019 1.2· 1020
Deep-Att + PosUnk Ensemble [32] 40.4 8.0· 1020
GNMT + RL Ensemble [31] 26.30 41.16 1.8· 1020 1.1· 1021
ConvS2S Ensemble [8] 26.36 41.29 7.7· 1019 1.2· 1021
Transformer (base model) 27.3 38.1 3.3 · 1018
Transformer (big) 28.4 41.0 2.3· 1019
Label Smoothing During training, we employed label smoothing of value ϵls = 0.1 [30]. This
hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
6 Results
6.1 Machine Translation
On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big)
```

### Retrieved Chunk 3

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 8
Chunk ID: 6971237f3f1b90b5e3e591764062f812

```text
on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014
English-to-French translation tasks, we achieve a new state of the art. In the former task our best
model outperforms even all previously reported ensembles.
We are excited about the future of attention-based models and plan to apply them to other tasks. We
plan to extend the Transformer to problems involving input and output modalities other than text and
to investigate local, restricted attention mechanisms to efﬁciently handle large inputs and outputs
such as images, audio and video. Making generation less sequential is another research goals of ours.
The code we used to train and evaluate our models is available at https://github.com/
tensorflow/tensor2tensor.
Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful
comments, corrections and inspiration.
9
```

### Retrieved Chunk 4

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 0
Chunk ID: f0099bb9fe1537824008b8fa53278a63

```text
be superior in quality while being more parallelizable and requiring signiﬁcantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-
to-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.0 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature.
1 Introduction
Recurrent neural networks, long short-term memory [12] and gated recurrent [7] neural networks
in particular, have been ﬁrmly established as state of the art approaches in sequence modeling and
transduction problems such as language modeling and machine translation [ 29, 2, 5]. Numerous
efforts have since continued to push the boundaries of recurrent language models and encoder-decoder
architectures [31, 21, 13].
```

## QUESTION Q14

### Question
Contrast the mathematical and architectural mechanisms by which RAG-Sequence and RAG-Token marginalize over latent retrieved documents during sequence generation.

### Retrieved Chunk 1

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 2
Chunk ID: bcb1e6f9a5a7e4f0bd66a284aaca55ca

```text
is marginalized to get the seq2seq probability p(y|x) via a top-K approximation. Concretely, the
top K documents are retrieved using the retriever, and the generator produces the output sequence
probability for each document, which are then marginalized,
pRAG-Sequence(y|x) ≈
∑
z∈top-k(p(·|x))
pη(z|x)pθ(y|x,z ) =
∑
z∈top-k(p(·|x))
pη(z|x)
N∏
i
pθ(yi|x,z,y 1:i−1)
RAG-Token Model In the RAG-Token model we can draw a different latent document for each
target token and marginalize accordingly. This allows the generator to choose content from several
documents when producing an answer. Concretely, the top K documents are retrieved using the
retriever, and then the generator produces a distribution for the next output token for each document,
before marginalizing, and repeating the process with the following output token, Formally, we deﬁne:
pRAG-Token(y|x) ≈
N∏
i
∑
z∈top-k(p(·|x))
pη(z|x)pθ(yi|x,z,y 1:i−1)
```

### Retrieved Chunk 2

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 2
Chunk ID: 5d4463bac5ab237a453733076f62b8c6

```text
byθ that generates a current token based on a context of the previousi− 1 tokensy1:i−1, the original
inputx and a retrieved passagez.
To train the retriever and generator end-to-end, we treat the retrieved document as a latent variable.
We propose two models that marginalize over the latent documents in different ways to produce a
distribution over generated text. In one approach, RAG-Sequence, the model uses the same document
to predict each target token. The second approach, RAG-Token, can predict each target token based
on a different document. In the following, we formally introduce both models and then describe the
pη andpθ components, as well as the training and decoding procedure.
2.1 Models
RAG-Sequence Model The RAG-Sequence model uses the same retrieved document to generate
the complete sequence. Technically, it treats the retrieved document as a single latent variable that
is marginalized to get the seq2seq probability p(y|x) via a top-K approximation. Concretely, the
```

### Retrieved Chunk 3

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 1
Chunk ID: e003fda3a482f1c12f05efbd3d8db1b1

```text
retriever (Dense Passage Retriever [26], henceforth DPR) provides latent documents conditioned on
the input, and the seq2seq model (BART [32]) then conditions on these latent documents together with
the input to generate the output. We marginalize the latent documents with a top-K approximation,
either on a per-output basis (assuming the same document is responsible for all tokens) or a per-token
basis (where different documents are responsible for different tokens). Like T5 [51] or BART, RAG
can be ﬁne-tuned on any seq2seq task, whereby both the generator and retriever are jointly learned.
There has been extensive previous work proposing architectures to enrich systems with non-parametric
memory which are trained from scratch for speciﬁc tasks, e.g. memory networks [ 64, 55], stack-
augmented networks [25] and memory layers [ 30]. In contrast, we explore a setting where both
parametric and non-parametric memory components are pre-trained and pre-loaded with extensive
```

### Retrieved Chunk 4

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 0
Chunk ID: 5797128399f0cc7dac99fb2beb0f86ea

```text
trained models with a differentiable access mechanism to explicit non-parametric
memory have so far been only investigated for extractive downstream tasks. We
explore a general-purpose ﬁne-tuning recipe for retrieval-augmented generation
(RAG) — models which combine pre-trained parametric and non-parametric mem-
ory for language generation. We introduce RAG models where the parametric
memory is a pre-trained seq2seq model and the non-parametric memory is a dense
vector index of Wikipedia, accessed with a pre-trained neural retriever. We com-
pare two RAG formulations, one which conditions on the same retrieved passages
across the whole generated sequence, and another which can use different passages
per token. We ﬁne-tune and evaluate our models on a wide range of knowledge-
intensive NLP tasks and set the state of the art on three open domain QA tasks,
outperforming parametric seq2seq models and task-speciﬁc retrieve-and-extract
```

## QUESTION Q15

### Question
Differentiate between the root causes targeted by OWASP Top Ten Category A04 (Insecure Design) and Category A08 (Software and Data Integrity Failures), explaining how their lifecycle positions and threat vectors differ.

### Retrieved Chunk 1

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 11
Chunk ID: 82a594af525a7c323beafd2502418796

```text
A08: Software & Data Integrity Failures
“Software and data integrity failures relate to code and infrastructure that does not 
protect against integrity violations. An example of this is where an application 
relies upon plugins, libraries, or modules from untrusted sources, repositories, and 
content delivery networks (CDNs).” *from the OWASP Top Ten page
Sounds like Software Supply Chain but really isn’t
Difference is that these vulnerabilities leverage execution within. 
This groups vulnerabilities injected into the payloads of otherwise benign artifacts 
before endpoint delivery  - for example
Insecure Deserialization - the log4j vulnerability this week
```

### Retrieved Chunk 2

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 10
Chunk ID: 3398a24855d499066b48e5a04796f008

```text
A04: Insecure Design
You didn’t Shift-Left far enough
Secure Design and Secure Patterns
Threat Modeling
But we already do SAST/DAST
Before the code 
Context is Everything
```

### Retrieved Chunk 3

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 9
Chunk ID: f12421ff1be3fb78852db911efd5806b

```text
New to the Top Ten
● Insecure Design
● Software & Data Integrity Failures
● Server Side Request Forgery
```

### Retrieved Chunk 4

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 14
Chunk ID: 010d15e4dc20eb266ea72418282ddfd2

```text
Use Case OWASP Top 10 2021 OWASP Application Security Verification Standard
Awareness Yes 
Training Entry level Comprehensive
Design and architecture Occasionally Yes
Coding standard Bare minimum Yes
Secure Code review Bare minimum Yes
Peer review checklist Bare minimum Yes
Unit testing Occasionally Yes
Integration testing Occasionally Yes
Penetration testing Bare minimum Yes
Tool support Bare minimum Yes
Secure Supply Chain Occasionally Yes
```

## QUESTION Q16

### Question
Compare the computational complexity per layer, minimum number of sequential operations, and maximum path lengths of Self-Attention layers against Recurrent and Convolutional layer types.

### Retrieved Chunk 1

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 5
Chunk ID: 64e6d0b26400b4e88129c50fe2f26d20

```text
and output sequences, the easier it is to learn long-range dependencies [11]. Hence we also compare
the maximum path length between any two input and output positions in networks composed of the
different layer types.
As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially
executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of
computational complexity, self-attention layers are faster than recurrent layers when the sequence
length n is smaller than the representation dimensionality d, which is most often the case with
sentence representations used by state-of-the-art models in machine translations, such as word-piece
[31] and byte-pair [25] representations. To improve computational performance for tasks involving
very long sequences, self-attention could be restricted to considering only a neighborhood of sizer in
6
```

### Retrieved Chunk 2

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 5
Chunk ID: 99d8720534ef95ba17b506313109687e

```text
tional layers commonly used for mapping one variable-length sequence of symbol representations
(x1,...,x n) to another sequence of equal length (z1,...,z n), with xi,z i∈ Rd, such as a hidden
layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we
consider three desiderata.
One is the total computational complexity per layer. Another is the amount of computation that can
be parallelized, as measured by the minimum number of sequential operations required.
The third is the path length between long-range dependencies in the network. Learning long-range
dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the
ability to learn such dependencies is the length of the paths forward and backward signals have to
traverse in the network. The shorter these paths between any combination of positions in the input
and output sequences, the easier it is to learn long-range dependencies [11]. Hence we also compare
```

### Retrieved Chunk 3

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 6
Chunk ID: 5d6610d9fed6fd312b8646fb6cc1cd67

```text
the input sequence centered around the respective output position. This would increase the maximum
path length toO(n/r). We plan to investigate this approach further in future work.
A single convolutional layer with kernel widthk<n does not connect all pairs of input and output
positions. Doing so requires a stack ofO(n/k) convolutional layers in the case of contiguous kernels,
orO(logk(n)) in the case of dilated convolutions [ 15], increasing the length of the longest paths
between any two positions in the network. Convolutional layers are generally more expensive than
recurrent layers, by a factor of k. Separable convolutions [ 6], however, decrease the complexity
considerably, toO(k·n·d +n·d2). Even with k = n, however, the complexity of a separable
convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer,
the approach we take in our model.
```

### Retrieved Chunk 4

Source: NIPS-2017-attention-is-all-you-need-Paper.pdf
Page: 1
Chunk ID: d9da888007a1ce0f7ca704eaed7aa186

```text
Recurrent models typically factor computation along the symbol positions of the input and output
sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden
statesht, as a function of the previous hidden stateht−1 and the input for positiont. This inherently
sequential nature precludes parallelization within training examples, which becomes critical at longer
sequence lengths, as memory constraints limit batching across examples. Recent work has achieved
signiﬁcant improvements in computational efﬁciency through factorization tricks [18] and conditional
computation [26], while also improving model performance in case of the latter. The fundamental
constraint of sequential computation, however, remains.
Attention mechanisms have become an integral part of compelling sequence modeling and transduc-
tion models in various tasks, allowing modeling of dependencies without regard to their distance in
```

## QUESTION Q17

### Question
Differentiate between the structural definition and operational purpose of "CSF Organizational Profiles" and "CSF Tiers" as defined in the NIST CSF 2.0.

### Retrieved Chunk 1

Source: NIST.CSWP.29.pdf
Page: 10
Chunk ID: cce869a7ea4090be9710d7c026f61152

```text
NIST CSWP 29  The NIST Cybersecurity Framework (CSF) 2.0 
February 26, 2024 
  6 
3. Introduction to CSF Profiles and Tiers 
This section defines the concepts of CSF Profiles and Tiers. 
3.1. CSF Profiles 
A CSF Organizational Profile describes an organization’s current and/or target cybersecurity 
posture in terms of the Core’s outcomes. Organizational Profiles are used to understand, tailor, 
assess, prioritize, and communicate the Core’s outcomes by considering an organization’s 
mission objectives, stakeholder expectations, threat landscape, and requirements. An 
organization can then prioritize its actions to achieve specific outcomes and communicate that 
information to stakeholders.  
Every Organizational Profile includes one or both of the following: 
1. A Cur
rent Profile specifies the Core outcomes that an organization is currently achieving 
(or attempting to achieve) and characterizes how or to what extent each outcome is 
being achieved.
```

### Retrieved Chunk 2

Source: NIST.CSWP.29.pdf
Page: 12
Chunk ID: 2062018a71b19290d92dd166c618c1d1

```text
The NIST CSF website provides additional information on using Profiles a nd Tiers. It includes 
pointers to NIST-hosted O rganizational Profile templates and a repository of Community 
Profiles in a variety of machine-read able and human-usable formats.
```

### Retrieved Chunk 3

Source: NIST.CSWP.29.pdf
Page: 30
Chunk ID: 8f9df0918bc06f6c6b156131202477bc

```text
CSF Quick Start Guide 
A supplementary resource that gives brief, actionable guidance on specific CSF-related topics. 
CSF Subcategory 
A group of more specific outcomes of technical and management cybersecurity activities that comprise a CSF 
Category. 
CSF Target Profile 
A part of an Organizational Profile that specifies the desired Core outcomes that an organization has selected and 
prioritized for achieving its cybersecurity risk management objectives. 
CSF Tier 
A characterization of the rigor of an organization’s cybersecurity risk governance and management practices. 
There are four Tiers: Partial (Tier 1), Risk Informed (Tier 2), Repeatable (Tier 3), and Adaptive (Tier 4).
```

### Retrieved Chunk 4

Source: NIST.CSWP.29.pdf
Page: 10
Chunk ID: bc73366b53816b56f2f84b89193f14d7

```text
(or attempting to achieve) and characterizes how or to what extent each outcome is 
being achieved.  
2. A Target Profile specifies the desired outcomes that an organization has selected and 
prioritized for achieving its cybersecurity risk management objectives. A Target Profile 
considers anticipated changes to the organization’s cybersecurity posture, such as new 
requirements, new technology adoption, and threat intelligence trends. 
 
A Community P rofile is a baseline of CSF outcomes that is created and published to address 
shared interests and goals among a number of organizations. A Community Profile is 
typically developed for a particular sector, subsector, technology, threat type, or other use 
case. An organization can use a Community Profile as the basis for its own Target Profile. 
Examples of Community Profiles can be found on the NIST CSF website.  
The
 steps shown in Fig. 3 and summarized below illustrate one way that an organization could
```

## QUESTION Q18

### Question
Contrast RAG's dense retriever (DPR) with a traditional word overlap-based BM25 retriever based on task evaluation performance, and identify which specific task BM25 outperforms DPR in and why.

### Retrieved Chunk 1

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 6
Chunk ID: 1bd602768374dedad995fa163db187ac

```text
p(z|x). Table 6 shows the results. For FEVER, BM25 performs best, perhaps since FEVER claims are
heavily entity-centric and thus well-suited for word overlap-based retrieval. Differentiable retrieval
improves results on all other tasks, especially for Open-Domain QA, where it is crucial.
Index hot-swapping An advantage of non-parametric memory models like RAG is that knowledge
can be easily updated at test time. Parametric-only models like T5 or BART need further training to
update their behavior as the world changes. To demonstrate, we build an index using the DrQA [5]
Wikipedia dump from December 2016 and compare outputs from RAG using this index to the newer
index from our main results (December 2018). We prepare a list of 82 world leaders who had changed
7
```

### Retrieved Chunk 2

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 4
Chunk ID: e4bf84824fcca9928c3aa88cd3e0207f

```text
4 Results
4.1 Open-domain Question Answering
Table 1 shows results for RAG along with state-of-the-art models. On all four open-domain QA
tasks, RAG sets a new state of the art (only on the T5-comparable split for TQA). RAG combines
the generation ﬂexibility of the “closed-book” (parametric only) approaches and the performance of
"open-book" retrieval-based approaches. Unlike REALM and T5+SSM, RAG enjoys strong results
without expensive, specialized “salient span masking” pre-training [ 20]. It is worth noting that RAG’s
retriever is initialized using DPR’s retriever, which uses retrieval supervision on Natural Questions
and TriviaQA. RAG compares favourably to the DPR QA system, which uses a BERT-based “cross-
encoder” to re-rank documents, along with an extractive reader. RAG demonstrates that neither a
re-ranker nor extractive reader is necessary for state-of-the-art performance.
There are several advantages to generating answers even when it is possible to extract them. Docu-
```

### Retrieved Chunk 3

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 6
Chunk ID: 4d2129ba789ba936494526efb19fb0fb

```text
BART for Jeopardy question generation. Following recent work on diversity-promoting decoding
[33, 59, 39], we also investigate generation diversity by calculating the ratio of distinct ngrams to
total ngrams generated by different models. Table 5 shows that RAG-Sequence’s generations are
more diverse than RAG-Token’s, and both are signiﬁcantly more diverse than BART without needing
any diversity-promoting decoding.
Retrieval Ablations A key feature of RAG is learning to retrieve relevant information for the task.
To assess the effectiveness of the retrieval mechanism, we run ablations where we freeze the retriever
during training. As shown in Table 6, learned retrieval improves results for all tasks.
We compare RAG’s dense retriever to a word overlap-based BM25 retriever [53]. Here, we replace
RAG’s retriever with a ﬁxed BM25 system, and use BM25 retrieval scores as logits when calculating
p(z|x). Table 6 shows the results. For FEVER, BM25 performs best, perhaps since FEVER claims are
```

### Retrieved Chunk 4

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 1
Chunk ID: e003fda3a482f1c12f05efbd3d8db1b1

```text
retriever (Dense Passage Retriever [26], henceforth DPR) provides latent documents conditioned on
the input, and the seq2seq model (BART [32]) then conditions on these latent documents together with
the input to generate the output. We marginalize the latent documents with a top-K approximation,
either on a per-output basis (assuming the same document is responsible for all tokens) or a per-token
basis (where different documents are responsible for different tokens). Like T5 [51] or BART, RAG
can be ﬁne-tuned on any seq2seq task, whereby both the generator and retriever are jointly learned.
There has been extensive previous work proposing architectures to enrich systems with non-parametric
memory which are trained from scratch for speciﬁc tasks, e.g. memory networks [ 64, 55], stack-
augmented networks [25] and memory layers [ 30]. In contrast, we explore a setting where both
parametric and non-parametric memory components are pre-trained and pre-loaded with extensive
```

## QUESTION Q19

### Question
How does the NIST CSF 2.0 describe the conceptual relationship, overlap, and practical distinctions between "Cybersecurity Risk" and "Privacy Risk"?

### Retrieved Chunk 1

Source: NIST.CSWP.29.pdf
Page: 17
Chunk ID: fe84c42e1260adef275decbbf0f98574

```text
The NIST Privacy Framework a
nd Cybersecurity Framework can be used together to 
address the different aspects of cybersecurity and privacy risks. Additionally, NIST’s 
Privacy Risk Assessment Methodology (PRAM) h as a catalog of example problems for 
use in privacy risk assessments. 
• Suppl y chain risk s: An organization can use the CSF to foster cybersecurity risk oversight 
and communications with stakeholders across supply chains. All types of technology rely 
on a complex, globally distributed, extensive, and interconnected supply chain 
ecosystem with geographically diverse routes and multiple levels of outsourcing. This 
ecosystem is composed of public- and private-sector entities (e.g., acquirers, suppliers, 
developers, system integrators, external system service providers, and other 
technology-related service providers) that interact to research, develop, design, 
manufacture, acquire, deliver, integrate, operate, maintain, dispose of, and otherwise
```

### Retrieved Chunk 2

Source: NIST.CSWP.29.pdf
Page: 17
Chunk ID: e304ee4b6b0b5d8a6cc1ac3c1ed66127

```text
NIST CSWP 29  The NIST Cybersecurity Framework (CSF) 2.0 
February 26, 2024 
  13 
 
Fig. 6. Cybersecurity and privacy risk relationship 
Cybersecurity risk management is essential for addressing privacy risks related to the 
loss of the confidentiality, integrity, and availability of individuals’ data. For example, 
data breaches could lead to identity theft. However, privacy risks can also arise by 
means that are unrelated to cybersecurity incidents.  
An organization processes data to achieve mission or business purposes, which can 
sometimes give rise to privacy events whereby individuals may experience problems as 
a result of the data processing. These problems can be expressed in various ways, but 
NIST describes them as ranging from dignity-type effects (e.g., embarrassment or 
stigma) to more tangible harms (e.g., discrimination, economic loss, or physical harm). 
The NIST Privacy Framework a
nd Cybersecurity Framework can be used together to
```

### Retrieved Chunk 3

Source: NIST.CSWP.29.pdf
Page: 1
Chunk ID: 2fada1c0ee7b130c9cd516016c096e5e

```text
T he NIST Cybersecurity Framework (CSF) 2.0 
 i 
NIST CSWP 29 
February 26, 2024 
Abstract
T
he NIST Cybersecurity Framework (CSF) 2.0 provides guidance to industry, government 
agencies, and other organizations to manage cybersecurity risks. It offers a taxonomy of high-
level cybersecurity outcomes that can be used by any organization — regardless of its size, 
sector, or maturity — to better understand, assess, prioritize, and communicate its 
cybersecurity efforts. The CSF does not prescribe how outcomes should be achieved. Rather, it 
links to online resources that provide additional guidance on practices and controls that could 
be used to achieve those outcomes. This document describes CSF 2.0, its components, and 
some of the many ways that it can be used.  
Keywords 
cybersecurity; Cybersecurity Framework (CSF); cybersecurity risk governance; cybersecurity risk 
management; enterprise risk management; Profiles; Tiers. 
Audience
```

### Retrieved Chunk 4

Source: NIST.CSWP.29.pdf
Page: 3
Chunk ID: c0ccac9678a4e64d9da78160b9eb4ebc

```text
Fig. 3. Steps for creating and using a CSF Organizational Profile ...........................................................6  
Fig. 4. CSF Tiers for cybersecurity risk governance and management ...................................................8  
Fig. 5. Using the CSF to improve risk management communication .................................................... 10 
Fig. 6. Cybersecurity and privacy risk relationship ............................................................................. 13
```

## QUESTION Q20

### Question
What are the specific security configurations, operating system parameters, and prescriptive access control list (ACL) rules mandated by the NIST CSF 2.0 to protect endpoints from ransomware?

### Retrieved Chunk 1

Source: NIST.CSWP.29.pdf
Page: 1
Chunk ID: 2fada1c0ee7b130c9cd516016c096e5e

```text
T he NIST Cybersecurity Framework (CSF) 2.0 
 i 
NIST CSWP 29 
February 26, 2024 
Abstract
T
he NIST Cybersecurity Framework (CSF) 2.0 provides guidance to industry, government 
agencies, and other organizations to manage cybersecurity risks. It offers a taxonomy of high-
level cybersecurity outcomes that can be used by any organization — regardless of its size, 
sector, or maturity — to better understand, assess, prioritize, and communicate its 
cybersecurity efforts. The CSF does not prescribe how outcomes should be achieved. Rather, it 
links to online resources that provide additional guidance on practices and controls that could 
be used to achieve those outcomes. This document describes CSF 2.0, its components, and 
some of the many ways that it can be used.  
Keywords 
cybersecurity; Cybersecurity Framework (CSF); cybersecurity risk governance; cybersecurity risk 
management; enterprise risk management; Profiles; Tiers. 
Audience
```

### Retrieved Chunk 2

Source: NIST.CSWP.29.pdf
Page: 19
Chunk ID: 04247d38fe3bf1cab46b1461a41805cf

```text
2.0 website a nd through the CSF 2.0 Reference Tool, which allows users to explore t hem and 
export them in human- and machine-readable formats. The CSF 2.0 Core is also available in a 
legacy format s imilar to that of CSF 1.1. 
  
Function Category Category Identifier 
Govern (GV) Organizational Context GV.OC 
 Risk Management Strategy GV.RM 
 Roles, Responsibilities, and Authorities GV.RR 
 Policy GV.PO 
 Oversight GV.OV 
 Cybersecurity Supply Chain Risk Management GV.SC 
Identify (ID) Asset Management ID.AM 
 Risk Assessment ID.RA 
 Improvement ID.IM 
Protect (PR) Identity Management, Authentication, and Access Control PR.AA 
 Awareness and Training PR.AT 
 Data Security PR.DS 
 Platform Security PR.PS 
 Technology Infrastructure Resilience PR.IR 
Detect (DE) Continuous Monitoring  DE.CM  
 Adverse Event Analysis  DE.AE  
Respond (RS) Incident Management RS.MA 
 Incident Analysis RS.AN 
 Incident Response Reporting and Communication RS.CO 
 Incident Mitigation RS.MI
```

### Retrieved Chunk 3

Source: NIST.CSWP.29.pdf
Page: 13
Chunk ID: 1ede3c3ea164967024e1561849d0e189

```text
document, develop, perform, monitor, analyze, assess, and exercise. The Examples are not a 
comprehensive list of all actions that could be taken by an organization to achieve an outcome, 
nor do they represent a baseline of required actions to address cybersecurity risks.  
Quick-Start Guides (QSGs) are brief documents on specific CSF-related topics and are often 
tailored to specific audiences. QSGs can help an organization implement the CSF because they 
distill specific portions of the CSF into actionable “first steps” that an organization can consider 
on the path to improving their cybersecurity posture and management of associated risks. The 
guides are revised in their own time frames, and new guides are added as needed.  
Suggestions for new Informative References for CSF 2.0 can always be shared with NIST at 
olir@nist.gov. Suggestions for other resources to reference on the NIST CSF website, including 
additional QSG topics, should be directed to cyberframework@nist.gov.
```

### Retrieved Chunk 4

Source: NIST.CSWP.29.pdf
Page: 24
Chunk ID: aca5e309d51996f6c91ecefc183c36bd

```text
that they possess the knowledge and skills to perform relevant tasks with cybersecurity 
risks in mind 
• D ata Security (PR.DS): Data are managed consistent with the organization’s risk strategy to 
protect the confidentiality, integrity, and availability of information 
o P R.DS- 01: The confidentiality, integrity, and availability of data-at-rest are protected 
o PR.DS-02: The confidentiality, integrity, and availability of data-in-transit are protected 
o PR.DS-10: The confidentiality, integrity, and availability of data-in-use are protected 
o PR.DS-11: Backups of data are created, protected, maintained, and tested 
• P latform Security (PR.PS): The hardware, software (e.g., firmware, operating systems, 
applications), and services of physical and virtual platforms are managed consistent with 
the organization’s risk strategy to protect their confidentiality, integrity, and availability 
o PR .PS -01: Configuration management practices are established and applied
```

## QUESTION Q21

### Question
What are the specific mathematical parameters and weight training updates applied to RAG's Wikipedia document encoder (BERT_d) during retriever fine-tuning?

### Retrieved Chunk 1

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 3
Chunk ID: 0ed20846f77a7fd7cee94022a8713484

```text
minimize the negative marginal log-likelihood of each target,∑
j− logp(yj|xj) using stochastic
gradient descent with Adam [28]. Updating the document encoder BERTd during training is costly as
it requires the document index to be periodically updated as REALM does during pre-training [20].
We do not ﬁnd this step necessary for strong performance, and keep the document encoder (and
index) ﬁxed, only ﬁne-tuning the query encoder BERTq and the BART generator.
2.5 Decoding
At test time, RAG-Sequence and RAG-Token require different ways to approximatearg maxyp(y|x).
RAG-Token The RAG-Token model can be seen as a standard, autoregressive seq2seq genera-
tor with transition probability: p′
θ(yi|x,y 1:i−1) = ∑
z∈top-k(p(·|x))pη(zi|x)pθ(yi|x,zi,y 1:i−1) To
decode, we can plugp′
θ(yi|x,y 1:i−1) into a standard beam decoder.
RAG-Sequence For RAG-Sequence, the likelihoodp(y|x) does not break into a conventional per-
```

### Retrieved Chunk 2

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 17
Chunk ID: 2f35dbe1b36046065203093ad24253c1

```text
logit by learning (i) a document embedding for the null document, (ii) a static learnt bias term, or
(iii) a neural network to predict the logit. We did not ﬁnd that these improved performance, so in
the interests of simplicity, we omit them. For Open MS-MARCO, where useful retrieved documents
cannot always be retrieved, we observe that the model learns to always retrieve a particular set of
documents for questions that are less likely to beneﬁt from retrieval, suggesting that null document
mechanisms may not be necessary for RAG.
G Parameters
Our RAG models contain the trainable parameters for the BERT-base query and document encoder of
DPR, with 110M parameters each (although we do not train the document encoder ourselves) and
406M trainable parameters from BART-large, 406M parameters, making a total of 626M trainable
18
```

### Retrieved Chunk 3

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 3
Chunk ID: 4ff8b852536f95845177d8e7a398e1a0

```text
we can make a further approximation thatpθ(y|x,zi)≈ 0 wherey was not generated during beam
search fromx,zi. This avoids the need to run additional forward passes once the candidate set Y has
been generated. We refer to this decoding procedure as “Fast Decoding.”
3 Experiments
We experiment with RAG in a wide range of knowledge-intensive tasks. For all experiments, we use
a single Wikipedia dump for our non-parametric knowledge source. Following Lee et al. [31] and
Karpukhin et al. [26], we use the December 2018 dump. Each Wikipedia article is split into disjoint
100-word chunks, to make a total of 21M documents. We use the document encoder to compute an
embedding for each document, and build a single MIPS index using FAISS [23] with a Hierarchical
Navigable Small World approximation for fast retrieval [37]. During training, we retrieve the top
k documents for each query. We considerk∈{ 5, 10} for training and setk for test time using dev
```

### Retrieved Chunk 4

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 0
Chunk ID: 5797128399f0cc7dac99fb2beb0f86ea

```text
trained models with a differentiable access mechanism to explicit non-parametric
memory have so far been only investigated for extractive downstream tasks. We
explore a general-purpose ﬁne-tuning recipe for retrieval-augmented generation
(RAG) — models which combine pre-trained parametric and non-parametric mem-
ory for language generation. We introduce RAG models where the parametric
memory is a pre-trained seq2seq model and the non-parametric memory is a dense
vector index of Wikipedia, accessed with a pre-trained neural retriever. We com-
pare two RAG formulations, one which conditions on the same retrieved passages
across the whole generated sequence, and another which can use different passages
per token. We ﬁne-tune and evaluate our models on a wide range of knowledge-
intensive NLP tasks and set the state of the art on three open domain QA tasks,
outperforming parametric seq2seq models and task-speciﬁc retrieve-and-extract
```

## QUESTION Q23

### Question
Provide the detailed, step-by-step secure coding checklist and unit testing suite recommended by OWASP to prevent Server-Side Request Forgery (SSRF) vulnerabilities.

### Retrieved Chunk 1

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 14
Chunk ID: 010d15e4dc20eb266ea72418282ddfd2

```text
Use Case OWASP Top 10 2021 OWASP Application Security Verification Standard
Awareness Yes 
Training Entry level Comprehensive
Design and architecture Occasionally Yes
Coding standard Bare minimum Yes
Secure Code review Bare minimum Yes
Peer review checklist Bare minimum Yes
Unit testing Occasionally Yes
Integration testing Occasionally Yes
Penetration testing Bare minimum Yes
Tool support Bare minimum Yes
Secure Supply Chain Occasionally Yes
```

### Retrieved Chunk 2

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 12
Chunk ID: ec5a3ee6d34c8c690d9d6c9473654457

```text
A10: Server Side Request Forgery
● App fetches remote resource without validating URL supplied by user
● Survey-generated entry
○ Data not supporting – yet
● So what? Attackers can use SSRF to:
○ Scan for open ports on the network
○ Access files local to the server
○ Read metadata of cloud services
○ Abuse internal services for further mischief
● Plan is to roll into a category eventually – where would it make sense?
```

### Retrieved Chunk 3

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 15
Chunk ID: 7b41335da05336246bdb449fe116fb2a

```text
How to use the list?
● Evolution of the list has changed. How should we use it?
○ Less specific over time
○ OWASP Top Ten Vulnerabilities doesn’t mean the same any more
■ SAST/DAST Filters and how many results does this mean
■ Standards or Frameworks that say no OWASP Top Ten
```

### Retrieved Chunk 4

Source: 20211216_OWASP-MSP_OWASP_Top_Ten_2021.pdf
Page: 10
Chunk ID: 3398a24855d499066b48e5a04796f008

```text
A04: Insecure Design
You didn’t Shift-Left far enough
Secure Design and Secure Patterns
Threat Modeling
But we already do SAST/DAST
Before the code 
Context is Everything
```

## QUESTION Q24

### Question
Describe the mathematical formulas and loss parameters used to configure the "Null Document" probability mechanism in the final evaluated RAG-Sequence model.

### Retrieved Chunk 1

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 17
Chunk ID: 2f35dbe1b36046065203093ad24253c1

```text
logit by learning (i) a document embedding for the null document, (ii) a static learnt bias term, or
(iii) a neural network to predict the logit. We did not ﬁnd that these improved performance, so in
the interests of simplicity, we omit them. For Open MS-MARCO, where useful retrieved documents
cannot always be retrieved, we observe that the model learns to always retrieve a particular set of
documents for questions that are less likely to beneﬁt from retrieval, suggesting that null document
mechanisms may not be necessary for RAG.
G Parameters
Our RAG models contain the trainable parameters for the BERT-base query and document encoder of
DPR, with 110M parameters each (although we do not train the document encoder ourselves) and
406M trainable parameters from BART-large, 406M parameters, making a total of 626M trainable
18
```

### Retrieved Chunk 2

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 2
Chunk ID: bcb1e6f9a5a7e4f0bd66a284aaca55ca

```text
is marginalized to get the seq2seq probability p(y|x) via a top-K approximation. Concretely, the
top K documents are retrieved using the retriever, and the generator produces the output sequence
probability for each document, which are then marginalized,
pRAG-Sequence(y|x) ≈
∑
z∈top-k(p(·|x))
pη(z|x)pθ(y|x,z ) =
∑
z∈top-k(p(·|x))
pη(z|x)
N∏
i
pθ(yi|x,z,y 1:i−1)
RAG-Token Model In the RAG-Token model we can draw a different latent document for each
target token and marginalize accordingly. This allows the generator to choose content from several
documents when producing an answer. Concretely, the top K documents are retrieved using the
retriever, and then the generator produces a distribution for the next output token for each document,
before marginalizing, and repeating the process with the following output token, Formally, we deﬁne:
pRAG-Token(y|x) ≈
N∏
i
∑
z∈top-k(p(·|x))
pη(z|x)pθ(yi|x,z,y 1:i−1)
```

### Retrieved Chunk 3

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 3
Chunk ID: 0ed20846f77a7fd7cee94022a8713484

```text
minimize the negative marginal log-likelihood of each target,∑
j− logp(yj|xj) using stochastic
gradient descent with Adam [28]. Updating the document encoder BERTd during training is costly as
it requires the document index to be periodically updated as REALM does during pre-training [20].
We do not ﬁnd this step necessary for strong performance, and keep the document encoder (and
index) ﬁxed, only ﬁne-tuning the query encoder BERTq and the BART generator.
2.5 Decoding
At test time, RAG-Sequence and RAG-Token require different ways to approximatearg maxyp(y|x).
RAG-Token The RAG-Token model can be seen as a standard, autoregressive seq2seq genera-
tor with transition probability: p′
θ(yi|x,y 1:i−1) = ∑
z∈top-k(p(·|x))pη(zi|x)pθ(yi|x,zi,y 1:i−1) To
decode, we can plugp′
θ(yi|x,y 1:i−1) into a standard beam decoder.
RAG-Sequence For RAG-Sequence, the likelihoodp(y|x) does not break into a conventional per-
```

### Retrieved Chunk 4

Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf
Page: 16
Chunk ID: f9ed9ecb00d28d24245a2948a397a7c9

```text
Appendices for Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks
A Implementation Details
For Open-domain QA we report test numbers using 15 retrieved documents for RAG-Token models.
For RAG-Sequence models, we report test results using 50 retrieved documents, and we use the
Thorough Decoding approach since answers are generally short. We use greedy decoding for QA as
we did not ﬁnd beam search improved results. For Open-MSMarco and Jeopardy question generation,
we report test numbers using ten retrieved documents for both RAG-Token and RAG-Sequence,
and we also train a BART-large model as a baseline. We use a beam size of four, and use the Fast
Decoding approach for RAG-Sequence models, as Thorough Decoding did not improve performance.
B Human Evaluation
Figure 4: Annotation interface for human evaluation of factuality. A pop-out for detailed instructions
and a worked example appear when clicking "view tool guide".
```

## QUESTION Q25

### Question
Provide the prescriptive step-by-step compliance auditing checklist and timeline requirements an organization must execute to achieve a certified NIST CSF 2.0 Tier 4 (Adaptive) maturity rating.

### Retrieved Chunk 1

Source: NIST.CSWP.29.pdf
Page: 29
Chunk ID: edc1e88dd7dd46a5a91e9047ff613bca

```text
NIST CSWP 29  The NIST Cybersecurity Framework (CSF) 2.0 
February 26, 2024 
 
  25 
Tier Cybersecurity Risk Governance Cybersecurity Risk Management 
The organization risk strategy is informed by the 
cybersecurity risks associated with its suppliers and the 
products and services it acquires and uses. Personnel 
formally act upon those risks through mechanisms such 
as written agreements to communicate baseline 
requirements, governance structures (e.g., risk councils), 
and policy implementation and monitoring. These 
actions are implemented consistently and as intended 
and are continuously monitored and reviewed.  
Tier 4: 
Adaptive  
There is an organization-wide 
approach to managing cybersecurity 
risks that uses risk-informed policies, 
processes, and procedures to address 
potential cybersecurity events. The 
relationship between cybersecurity 
risks and organizational objectives is 
clearly understood and considered 
when making decisions. Executives
```

### Retrieved Chunk 2

Source: NIST.CSWP.29.pdf
Page: 1
Chunk ID: 2fada1c0ee7b130c9cd516016c096e5e

```text
T he NIST Cybersecurity Framework (CSF) 2.0 
 i 
NIST CSWP 29 
February 26, 2024 
Abstract
T
he NIST Cybersecurity Framework (CSF) 2.0 provides guidance to industry, government 
agencies, and other organizations to manage cybersecurity risks. It offers a taxonomy of high-
level cybersecurity outcomes that can be used by any organization — regardless of its size, 
sector, or maturity — to better understand, assess, prioritize, and communicate its 
cybersecurity efforts. The CSF does not prescribe how outcomes should be achieved. Rather, it 
links to online resources that provide additional guidance on practices and controls that could 
be used to achieve those outcomes. This document describes CSF 2.0, its components, and 
some of the many ways that it can be used.  
Keywords 
cybersecurity; Cybersecurity Framework (CSF); cybersecurity risk governance; cybersecurity risk 
management; enterprise risk management; Profiles; Tiers. 
Audience
```

### Retrieved Chunk 3

Source: NIST.CSWP.29.pdf
Page: 30
Chunk ID: 8f9df0918bc06f6c6b156131202477bc

```text
CSF Quick Start Guide 
A supplementary resource that gives brief, actionable guidance on specific CSF-related topics. 
CSF Subcategory 
A group of more specific outcomes of technical and management cybersecurity activities that comprise a CSF 
Category. 
CSF Target Profile 
A part of an Organizational Profile that specifies the desired Core outcomes that an organization has selected and 
prioritized for achieving its cybersecurity risk management objectives. 
CSF Tier 
A characterization of the rigor of an organization’s cybersecurity risk governance and management practices. 
There are four Tiers: Partial (Tier 1), Risk Informed (Tier 2), Repeatable (Tier 3), and Adaptive (Tier 4).
```

### Retrieved Chunk 4

Source: NIST.CSWP.29.pdf
Page: 29
Chunk ID: 537f5cb98ae8807a66b827f681359654

```text
and communicated.   
The organization adapts its cybersecurity practices 
based on previous and current cybersecurity activities, 
including lessons learned and predictive indicators. 
Through a process of continuous improvement that 
incorporates advanced cybersecurity technologies and 
practices, the organization actively adapts to a changing 
technological landscape and responds in a timely and 
effective manner to evolving, sophisticated threats. 
The organization uses real-time or near real-time 
information to understand and consistently act upon the 
cybersecurity risks associated with its suppliers and the 
products and services it acquires and uses. 
Cybersecurity information is constantly shared 
throughout the organization and with authorized third 
parties.
```
