# The Audio Auditor

paper link: [The Audio Auditor: Participant-Level Membership Inference in
Internet of Things Voice Services](https://arxiv.org/abs/1905.07082)

_ _ _
## Methodology
The primary task to train an audio auditor is to build up several shadow models to infer the targeted ASR model's decision boundary. We assume all learning algorithms Altar are known to the auditor; therefore, the learning algorithms for the shadow model are known accordingly (Al<sub>shd</sub> = Al<sub>tar</sub>). Different from the target model, we have full knowledge of the shadow models' ground truth. For a user *u* querying the model with her audio samples, if *u ∈ D<sub>shd</sub><sup>train</sup>*, we collapse the features extracted from these samples' results into one record and label it as "member"; otherwise, "nonmember". Taken all together with these labeled records (processed), a training dataset is set to train a binary classifier as the audit model using a supervised learning algorithm. As also evidenced in [[19]](), the more shadow models built, the more accurate the audit model performed.<br />

For participant-level membership, some users' pertinent characters are extracted from each output, including the transcription text (denoted as *TXT*), the posterior probability (denoted as *Probability*), and the audio frame length (denoted as *Frame Length*). The features of the auditor's training set are written as: `*{TXT1=type(string), Probability1=type(float), Frame_Length1=type(integer), ..., TXTn=type(txt), Probabilityn=type(float), Frame_Lengthn=type(integer), class}*`, where *n* is the number of audios belonging to a speaker.<br />
<br />

<p align="center"><img width="1009" alt="Audio Auditor Methodology" src="https://user-images.githubusercontent.com/13388819/67995768-573b9400-fca0-11e9-9114-3a2d1287dd8a.png"></p> 

_ _ _
## Build the ASR model
### Goals
Build up 3 ASR models, while one as the target model and the other two as the shadow models.

### Steps for each ASR model:
1. Install [pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi) toolkit

2. Separate TIMIT dataset into 3 subsets.

3. Follow pytorch-kaldi's instruction to build up the model using the TIMIT subset.

4. Gain the transcription results from the built ASR model and save it as .log in data/ folder

### Train an Auditor model
1. Preprocess:
```
~$ ./log2csv.sh
~$ python txt2csv.py
```

2. Build up the auditor model using the decision tree algoritghm:
```
~$ python audit_member.py
```

3. Plot the performance of the auditor model
```
~$ python plot_fig.py
```
