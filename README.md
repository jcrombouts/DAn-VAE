# DAn-VAE
Implementation of Disentangling Anomaly Detection Variational Autoencoder (DAn-VAE). I proposed this model in my Master thesis, which was the final step for the
master Data Science in Engineering at Eindhoven University of Technology. The thesis was rewarded with a 9.

If you are interested in my thesis, you can reach me by email: joris.rombouts@hotmail.com.

# Anomaly detection in image data sets using disentangled representations 
## Abstract
Visual surface inspection is important to save maintenance cost and maintain quality of products.
Visual surface inspection can be done by taking images of the product surface and analysing
them. Analysing images manually is time-consuming, prone to human error and requires domain
knowledge. Deep learning models can be used to automate this process and make it easier to scale.

This collection of images is high-dimensional, often contain few data points, is imbalanced
and only partly labelled. Furthermore, images contain randomness in factors that are distracting
for the task at hand. For example, the images can differ in illumination, background or weather
conditions that are not always possible to control during data collection. Next to this, there is
most of the time a small set of labeled samples explaining some common defect types available.
The defect types often are subtle patterns on product surfaces.

To address these challenges we formalise the visual surface inspection problem as a probabilistic
anomaly detection problem. Generative models are used to estimate the probability of an example
under a learnt distribution.

Previous works that utilise the small set of available anomalies still fail to address the understanding of random factors in the images. Generative models are sensitive to rare changes in image
factors, like brightness. Images containing rare factors are easily marked as anomalous because of
these rare factors. Using disentangled representation learning we aim to disentangle the anomaly
irrelevant from the anomaly relevant part in the image.

In this thesis we propose Disentangling Anomaly Detection Variational Autoencoder (DAnVAE), which learns a disentangled representation of the data to improve anomaly detection results. DAn-VAE divides the latent representation into two parts: anomaly relevant and irrelevant
information. We use a supervised anomaly detection technique that encourages DAn-VAE to
mostly use the anomaly relevant information for anomaly detection.

We evaluate our method on a public benchmark data set and a real-world anomaly data
set. Our results show that by disentangling brightness from the image the anomaly detection
performance can be improved in some cases. However, the disentanglement performance is very
sensitive to the model choice. Our experiments show that disentanglement is especially helpful for
smaller latent dimensions and dark images.