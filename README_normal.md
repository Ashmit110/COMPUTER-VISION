# COMPUTER VISION IMPLEMENTING VAE's

## Sources Referred

Stanford cs231n [youtube link](https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)

Variational Autoencoders-[link](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) [link](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

kl divergence-[link](https://encord.com/blog/kl-divergence-in-machine-learning/)

model architecture inspiration-[link1](https://colab.research.google.com/gist/rekalantar/2c303b69a11875dfba873aac11e44cfd/variationalautoencoder_pytorch.ipynb) [link2](https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac) [link3](https://youtu.be/VELQT1-hILo?si=AuS6bvygleaZaGPo)

## What i knew

To be honest never  heard of anything before the assignment.

## BLOG

### DAY 0(study)

watched the stanford lectures on VAE's and read the articles on variational encoders and kl divergence provided.

### DAY 1(implementation using conv architectue)(failed)

On this day i tried to implement Vaes on a convolution network,because convolution works better for image data type, i tried to merge the ideas in [link1](https://colab.research.google.com/gist/rekalantar/2c303b69a11875dfba873aac11e44cfd/variationalautoencoder_pytorch.ipynb) and [link2](https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac) but results where not satisfactory i spend the whole day trying to figure it out but results were not presentable.(can be found in "just_trying_convolutions folder")

### DAY 2(implementing using linear architecture)(succeed)

After a failed attempt i started to look up implemetations online and found [link3](https://youtu.be/VELQT1-hILo?si=AuS6bvygleaZaGPo) it worked great for the Mnist dataset. so i analysed the code (and also added few plotting function to judge the performance of my model)learned what mistakes i was making(what i think)-

1) convolutions are great for classification problems and have a tendency to classify data very far away in latent space from each other.
2) the dimention of the latent space that i was considering was also very high allowing huge gaps in the vector space.

(Note might try to fix my Conv architecture if time allows)

#### RUN1

lr=1e-4

epochs=300

![1709046415072](image/README_normal/1709046415072.png)

![1709046462076](image/README_normal/1709046462076.png)

![1709046499492](image/README_normal/1709046499492.png)

![1709046731701](image/README_normal/1709046731701.png)

Epoch 1         Average Loss:  232.33504851836395

Epoch 100       Average Loss:  142.18998690851942

Epoch 300       Average Loss:  137.17163253860602

KLD=-0.5*torch.sum(1+log_var-mean.pow(2) -log_var.exp())

coefficient of KLD (beta)

here, beta=0.5

OBSERVATION-THE Average loss decreases very negligibly after 100 epochs so using a lr deacay can help a lot .for the later runs i will run the model only for 100 epochs and finetune on that.

we can also see that model is having trouble in reconstructing 4 and 5 and regenerates 9 and 3 instead.From the latent space plot it is also obvious that model have a lot of overlap.TO FIX THIS changing the coefficient of KLD(here 0.5)would help since it acts as a weight that ensure the closesness/continuity of the data.

#### RUN 2

i set the coefficient of kld  as 0.4 and added a lr decay of 2e-7

lr=1e-4

beta=0.4

![1709048512165](image/README_normal/1709048512165.png)

![1709049150335](image/README_normal/1709049150335.png)

(FROM PREVIOUS RUN quality of reconstructed 4 has increases)

![1709048711083](image/README_normal/1709048711083.png)

![1709048737054](image/README_normal/1709048737054.png)

![1709048778699](image/README_normal/1709048778699.png)

Epoch 100       Average Loss:  141.8047986064013

observation

we can see that our Average Loss has decreased indicating that puting a learning rate decay was useful and also the quality of reconstructed 4 has also improved so changing beta was also a right decision.

#### RUN3












/

Epoch 500       Average Loss:  134.4445011216611

![1709035405468](image/eadmefinal/1709035405468.png)

![1709035463425](image/eadmefinal/1709035463425.png)

![1709036033640](image/eadmefinal/1709036033640.png)

Epoch 500       Average Loss:  134.4445011216611
