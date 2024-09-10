---
layout: distill
title: Epilepsy Detection Using Non-Linear Feature Analysis
description: A baseline setup
img: assets/img/EEG_featureextractor_proj/EEG_project_thumb.jpg
importance: 1
category: work
related_publications: true


_styles: >
  .fake-img {
    background: #ffffff;  /* Pure white background */
    border: none;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin-bottom: 16px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }
  .fake-img:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);
  }
  .fake-img p {
    font-family: 'Roboto', sans-serif;
    color: #333;
    text-align: center;
    margin: 16px 0;
    font-size: 18px;
    letter-spacing: 0.5px;
  }



bibliography: EEG_refs.bib

---

Electroencephalography (EEG) has long been an essential tool in neuroscience and clinical neurology for monitoring brain activity. With its non-invasive nature, EEG enables real-time recording of electrical signals produced by neurons in the brain, making it invaluable for diagnosing neurological conditions like epilepsy. However, the analysis of EEG signals can be challenging due to their non-stationary and complex nature. Researchers need efficient tools to extract meaningful patterns from these signals, especially for applications like automated seizure detection.

The main goal of this project is to extract features from epileptic data and perform classification to determine which types of features are most effective. As a baseline, we first extract statistical features from the time series and test how well they classify the data. Then, we apply the same classification tests using non-linear features. You can find all the project details on my GitHub repository: <a href="https://github.com/Cup-cake-lover/EEG_FeatureExtractor.git">EEG_FeatureExtractor</a>. 


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/EEG_featureextractor_proj/methodology.JPG" title="Methodology" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Basic methodology adapted from my original work presentation. The Schizophrenia part is for an another project article :P.
</div>


<h1> Aquiring data and Denoising </h1>


First, I will describe the basic methodology we will follow. The first step is to obtain a reliable dataset. For this project, I am using a standard dataset that is commonly available: the Bonn EEG dataset <d-cite key="PhysRevE.64.061907"></d-cite> . It is both easy to use and reliable, containing data with both normal and epileptic signatures. The dataset is divided into five categories: A, B, C, D, and E.



<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/EEG_featureextractor_proj/datadesc.png" title="description of data" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
   Description of the dataset.
</div>


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/EEG_featureextractor_proj/Normal_EEG.png" title="set A" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/EEG_featureextractor_proj/Normal_EEGB.png" title="set B" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/EEG_featureextractor_proj/Epileptic_EEG.png" title="set E" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Example EEG timeseries data from A (blue), B (green) and E (red)
</div>

To build a feature extractor we first parse our dataset. To do this, we first vectorize the data and save them into their respective arrays. Note that this is an implementation directly written to class named EpilepsyFeatureExtractor.

{% highlight python %}

def load_data(self):
        # List files in each folder
        for folder in self.folders:
            folder_path = os.path.join(self.base_path, folder)
            files = sorted(os.listdir(folder_path))
            self.data_names[folder] = [os.path.join(folder_path, f) for f in files if f.endswith('.eea')]

        # Load data from files
        for folder, arr in zip(self.folders, [self.arr_a, self.arr_b, self.arr_d, self.arr_e]):
            for file in self.data_names[folder]:
                data = pd.read_csv(file)
                data = np.array(data)
                arr = np.append(arr, data, axis=1)
            if folder == 'A':
                self.arr_a = arr
            elif folder == 'B':
                self.arr_b = arr
            elif folder == 'D':
                self.arr_d = arr
            elif folder == 'E':
                self.arr_e = arr
        # Apply denoising to the loaded data
        
        self.arr_a = self.denoise(self.arr_a.T)
        self.arr_b = self.denoise(self.arr_b.T)
        self.arr_d = self.denoise(self.arr_d.T)
        self.arr_e = self.denoise(self.arr_e.T)
{% endhighlight %}


Now that we have a dataset, we need to perform some sort of denoising. To acheive this, I tried using wavelet transform base approach. To this end, a recently developed method called Tunable Q wavelet transform is used by Selesnick et al <d-cite key="SELESNICK20112793"></d-cite>. since it is highly adaptable to differnet kind of timeseries profiles. I have adapted the implementation from 
 TQWT performs multiresolution analysis using a Q factor. After some trial and error, the Q and R parameters are set to be 6 and 5. 

Now that we have a dataset, the next step is to perform denoising. To achieve this, I used a wavelet transform-based approach. Specifically, I implemented a method called Tunable Q Wavelet Transform (TQWT), developed by Selesnick et al. This method is highly adaptable to different types of time series profiles. I adapted the implementation from <a href="https://github.com/jollyjonson/tqwt_tools.git">jollyjonson's TQWT tools</a>. TQWT performs multiresolution analysis using a Q factor, and after some trial and error, I set the Q and R parameters to 6 and 5, respectively.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/EEG_featureextractor_proj/subband.png" title="Decomposition" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Decomposition of an EEG timeseries into different subbands.
</div>

This is implemented in the class as,

{% highlight python %}

def denoise(self, arr):
        den_arr = np.empty((4096, 0))
        q = 6
        redundancy = 5
        stages = 10
        n = len(arr[0, :])
        for i in range(len(arr[:, 0])):
            x = arr[i, :]
            w = tqwt(x, q, redundancy, stages)
            y = itqwt(w, q, redundancy, n)
            y = np.array(y.real.reshape((4096, 1)))
            den_arr = np.append(den_arr, y, axis=1)
        return den_arr.T

{% endhighlight %}




After decomposition, the signal is recomposed to obtain the denoised version.



<h1> Extracting features </h1>

With the dataset denoised, the next step is to extract features from the data. We will focus on two sets of features: statistical and non-linear. Statistical features serve as a baseline, while non-linear features are the new additions we are exploring here.


<h2> Statistical features </h2>

To extract statistical features, we use the `MNE-features` package:


{% highlight python %}
from mne_features.feature_extraction import FeatureExtractor
from mne_features.feature_extraction import extract_features
{% endhighlight %}



We will analyze only a subset of the dataset, so I created a helper function to select the samples:

{% highlight python %}

def select_samples(self):
        print(self.arr_a.shape)
        data = np.array([self.arr_a[0:80], self.arr_b[0:80], self.arr_e[0:80]]).reshape(240, 1, 4096)
        arr = data.reshape(240, 4096)        
        return arr,data

{% endhighlight %}


Next, we create a function to extract the statistical features:

{% highlight python %}
def statistical_feature_extractor(self):
        _,data = self.select_samples()
        sfreq = 173.6 # Sampling frequency of the data
        selected_funcs = ['mean','std','kurtosis','skewness'] # Stastical features of the data vector
        stat_features = extract_features(data, sfreq, selected_funcs, funcs_params=None, n_jobs=1, ch_names=None, return_as_df=False) # Extracting them Using MNE package
        stat_features = stat_features.T
        
        return stat_features
{% endhighlight %}


Here, we extract four main statistical features: mean, standard deviation, kurtosis, and skewness. To visualize these features, I added a helper function:


{% highlight python %}
def plot_stat_features(self):
        features = self.statistical_feature_extractor()
        labels = ['Mean', 'Std', 'Kurtosis', 'Skewness']
        colors = ['black', 'red', 'blue', 'green']

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.ravel()

        for i in range(4):
            axs[i].plot(features[i, :],'o',color=colors[i], label=labels[i])
            axs[i].axvline(x=160,color='orange')
            axs[i].set_xlabel('Data Vector')
            axs[i].set_ylabel(labels[i])
            axs[i].legend()

        plt.tight_layout()
        plt.show()
{% endhighlight %}

It is important to note that the first 160 data points are from non-epileptic data (categories A and B). After this, the features are computed from epileptic data.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/EEG_featureextractor_proj/statfeatureresults.png" title="Statistical features" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Plotting statistical features.
</div>


By plotting the statistical features, we can see that the features computed for non-epileptic data differ significantly from those computed for epileptic data. However, the mean does not seem to be a useful feature for classification in this case.


<h2> Non-linear/Chaotic features </h2>

With the baseline established using statistical features, we now turn to non-linear features. Before diving into the implementation, let’s briefly outline the four non-linear features we will compute:

- Lyapunov exponent ($$ \lambda $$) 
- Hurst exponent ($$ H $$)
- Detrended Fluctuation Analysis ($$ DFA $$)
- Sample entropy ($$ S $$)

A brief description of them are given below

- Lyapunov Exponent ($$ \lambda $$): The Lyapunov exponent measures the rate at which nearby trajectories in a dynamical system diverge. A positive Lyapunov exponent indicates chaotic behavior, meaning that small changes in the initial conditions can lead to exponentially different outcomes. It is often used to quantify the presence of chaos in time series data.

- Hurst Exponent ($$ H $$): The Hurst exponent is a measure of the long-term memory of a time series. It quantifies the tendency of a time series to either regress strongly to the mean (H < 0.5), exhibit random walk behavior (H ≈ 0.5), or show persistent trends (H > 0.5). It is commonly used in fractal analysis and for understanding the predictability of time series.

- Detrended Fluctuation Analysis ($$ DFA $$): DFA is a technique used to detect long-range correlations in non-stationary time series. It removes trends from the data and then measures the fluctuation of the time series. This method is particularly useful for analyzing signals with varying trends or non-stationary properties, like EEG data.

- Sample Entropy ($$ S $$): Sample entropy quantifies the regularity and unpredictability of a time series. It measures the likelihood that two similar patterns in the time series remain similar over a given time interval. Lower sample entropy values indicate more predictability or regularity, while higher values suggest greater complexity or randomness.


All of these features can be computed using a single fucntional implementation in python. Here we use a python implemenation of these features using `nolds`. 


first we import nolds

{% highlight python %}
import nolds
{% endhighlight %}

{% highlight python %}
def chaotic_feature_extractor(self):
        data_s,_ = self.select_samples()
        Lyaps = []
        Hurst = []
        Entropy = []
        dfa = []

        for i in tqdm(range(data_s.shape[0])):
            l = nolds.lyap_r(data_s[i, :])  # Lyapunov Exponent
            h = nolds.hurst_rs(data_s[i, :])  # Hurst Exponent
            s = nolds.sampen(data_s[i, :])  # Sample entropy
            d = nolds.dfa(data_s[i, :])  # Detrended fluctuation analysis

            Lyaps.append(l)
            Hurst.append(h)
            Entropy.append(s)
            dfa.append(d)

        features = [Lyaps, Hurst, Entropy, dfa]
        return np.array(features).T
{% endhighlight %}



We also similarly create a function to plot the features w.r.t datavector index,

{% highlight python %}
def plot_chaotic_features(self):
        features = self.chaotic_feature_extractor()
        labels = ['Lyapunov Exponent', 'Hurst Exponent', 'Sample Entropy', 'DFA']
        colors = ['black', 'red', 'blue', 'green']

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.ravel()

        for i in range(4):
            axs[i].plot(features[:, i],'o',color=colors[i], label=labels[i])
            axs[i].axvline(x=160,color='orange')
            axs[i].set_xlabel('Data Vector')
            axs[i].set_ylabel(labels[i])
            axs[i].legend()

        plt.tight_layout()
        plt.show()
{% endhighlight %}



We get the following result,

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/EEG_featureextractor_proj/chaoticfeatureresults.png" title="Nonlinear features" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Plotting non-linear features.
</div>

Also we create function to plot the feature space, ie, plot two different features together. 

{% highlight python %}
def plot_2d_features(self):
        features = self.chaotic_feature_extractor()
        labels = ['Lyapunov Exponent', 'Hurst Exponent', 'Sample Entropy', 'DFA']
        colors = ['black', 'red', 'blue', 'green']

        pairs = list(itertools.combinations(range(len(labels)), 2))  # Generate all pairs of feature indices
        pair_labels = [(labels[x], labels[y]) for x, y in pairs]

        fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjust the number of subplots accordingly
        axs = axs.ravel()

        for i, (x, y) in enumerate(pairs):
            axs[i].scatter(features[:,x],features[:,y],label='Non Epileptic',alpha=0.7, edgecolors='black')
            axs[i].scatter(features[:,x][200:300],features[:,y][200:300],label='Epileptic',c='r', alpha=0.7, edgecolors='black')
            axs[i].set_xlabel(pair_labels[i][0])
            axs[i].set_ylabel(pair_labels[i][1])
            axs[i].set_title(f'{pair_labels[i][0]} vs {pair_labels[i][1]}')

        plt.tight_layout()
        plt.show()
{% endhighlight %}

And the resultant plot looks like this,


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/EEG_featureextractor_proj/twodfeatureplots.png" title="2D Feature plots" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Feature space plots
</div>

Ha! There is a clear seperation within the feature space. This is a very good sign! since it clearly says that we can classify them nicely.




<h1> Classification </h1>


Now to test classification, we use three classic classifiers. Support vector machines (SVM) K-nearest neighbours (KNN) and Random forest (RF). To achieve this, we first import all the neccessary classifiers from `scikit-learn` package.

{% highlight python %}
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from EpilepsyFeatureExtractor import EpilepsyFeatureExtractor
{% endhighlight %}

Since we have a limited amount of data, we employ cross-validation (CV) across different folds to ensure the robustness of our results. You can read more about cross-validation techniques <a href="https://scikit-learn.org/stable/modules/cross_validation.html">here</a>. For our classification task, we compute the CV scores and store them in a pandas DataFrame for easier tabular visualization.


{% highlight python %}
extractor = EpilepsyFeatureExtractor()
extractor.load_data()
#Extract features to perform the classification.
stat_features = extractor.statistical_feature_extractor()
chaotic_features = extractor.chaotic_feature_extractor()

# Create labels: 0 for non-epileptic, 1 for epileptic
y = np.zeros(240)
y[160:] = 1  # The last 80 samples are epileptic

def run_classifiers(features, y):
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        'SVM': svm.SVC(),
        'KNN': KNeighborsClassifier(n_neighbors=1)
    }

    results = []
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        pipe = Pipeline([('clf', clf)])
        scores = cross_val_score(pipe, features, y, cv=skf)
        results.append({
            'Classifier': name,
            'Mean Accuracy': np.mean(scores),
            'Std Dev': np.std(scores)
        })

    return results

# Run classifiers on statistical features
stat_results = run_classifiers(stat_features.T, y)

# Run classifiers on chaotic features
chaotic_results = run_classifiers(chaotic_features, y)

# Combine statistical and chaotic features
combined_features = np.hstack((stat_features.T, chaotic_features))

# Run classifiers on combined features
combined_results = run_classifiers(combined_features, y)

# Combine results and create a DataFrame
all_results = pd.DataFrame(stat_results + chaotic_results + combined_results)
all_results['Feature Type'] = ['Statistical']*len(stat_results) + ['Chaotic']*len(chaotic_results) + ['Combined']*len(combined_results)
# Display the results
print(all_results)
{% endhighlight %}





<h1> Results </h1>

The final scores are as follows,


| Classifier     |    Mean Accuracy   |   Std Dev   | Feature Type |
|----------------|:------------------:|------------:|--------------|
| RandomForest   |      0.995833       |    0.007217 | Statistical  |
| SVM            |      0.979167       |    0.021651 | Statistical  |
| KNN            |      1.000000       |    0.000000 | Statistical  |
| RandomForest   |      0.970833       |    0.007217 | Chaotic      |
| SVM            |      0.970833       |    0.007217 | Chaotic      |
| KNN            |      0.979167       |    0.013819 | Chaotic      |
| RandomForest   |      0.995833       |    0.007217 | Combined     |
| SVM            |      0.975000       |    0.018634 | Combined     |
| KNN            |      1.000000       |    0.000000 | Combined     |


We can see that the non-linear features provide comparable results in classification. While there are many classifiers to choose from, we always have the flexibility to introduce new features and combine them to achieve better classification accuracy.



<h1> Complete implementation </h1>

{% highlight python %}
class EpilepsyFeatureExtractor:
    def __init__(self):
        shutup.please()
        self.base_path = '/home/hari/projects/BonnData'
        self.folders = ['A', 'B', 'D', 'E']
        self.file_prefixes = {'A': 'Z', 'B': 'O', 'D': 'F', 'E': 'S'}
        self.data_names = {folder: [] for folder in self.folders}
        
        self.arr_a = np.empty((4096, 0))
        self.arr_b = np.empty((4096, 0))
        self.arr_d = np.empty((4096, 0))
        self.arr_e = np.empty((4096, 0))
        
    def load_data(self):
        # List files in each folder
        for folder in self.folders:
            folder_path = os.path.join(self.base_path, folder)
            files = sorted(os.listdir(folder_path))
            self.data_names[folder] = [os.path.join(folder_path, f) for f in files if f.endswith('.eea')]

        # Load data from files
        for folder, arr in zip(self.folders, [self.arr_a, self.arr_b, self.arr_d, self.arr_e]):
            for file in self.data_names[folder]:
                data = pd.read_csv(file)
                data = np.array(data)
                arr = np.append(arr, data, axis=1)
            if folder == 'A':
                self.arr_a = arr
            elif folder == 'B':
                self.arr_b = arr
            elif folder == 'D':
                self.arr_d = arr
            elif folder == 'E':
                self.arr_e = arr

        # Apply denoising to the loaded data
        self.arr_a = self.denoise(self.arr_a.T)
        self.arr_b = self.denoise(self.arr_b.T)
        self.arr_d = self.denoise(self.arr_d.T)
        self.arr_e = self.denoise(self.arr_e.T)
    
    def denoise(self, arr):
        den_arr = np.empty((4096, 0))
        q = 6
        redundancy = 5
        stages = 10
        n = len(arr[0, :])
        for i in range(len(arr[:, 0])):
            x = arr[i, :]
            w = tqwt(x, q, redundancy, stages)
            y = itqwt(w, q, redundancy, n)
            y = np.array(y.real.reshape((4096, 1)))
            den_arr = np.append(den_arr, y, axis=1)
        return den_arr.T
    
    def select_samples(self):
        print(self.arr_a.shape)
        data = np.array([self.arr_a[0:80], self.arr_b[0:80], self.arr_e[0:80]]).reshape(240, 1, 4096)
        arr = data.reshape(240, 4096)        
        return arr,data
    
    def statistical_feature_extractor(self):
        _,data = self.select_samples()
        sfreq = 173.6 # Sampling frequency of the data
        selected_funcs = ['mean','std','kurtosis','skewness'] # Stastical features of the data vector
        stat_features = extract_features(data, sfreq, selected_funcs, funcs_params=None, n_jobs=1, ch_names=None, return_as_df=False) # Extracting them Using MNE package
        stat_features = stat_features.T
        
        return stat_features
        
    
    def plot_stat_features(self):
        features = self.statistical_feature_extractor()
        labels = ['Mean', 'Std', 'Kurtosis', 'Skewness']
        colors = ['black', 'red', 'blue', 'green']

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.ravel()

        for i in range(4):
            axs[i].plot(features[i, :],'o',color=colors[i], label=labels[i])
            axs[i].axvline(x=160,color='orange')
            axs[i].set_xlabel('Data Vector')
            axs[i].set_ylabel(labels[i])
            axs[i].legend()

        plt.tight_layout()
        plt.show()
    
    
        
    
    def chaotic_feature_extractor(self):
        data_s,_ = self.select_samples()
        Lyaps = []
        Hurst = []
        Entropy = []
        dfa = []

        for i in tqdm(range(data_s.shape[0])):
            l = nolds.lyap_r(data_s[i, :])  # Lyapunov Exponent
            h = nolds.hurst_rs(data_s[i, :])  # Hurst Exponent
            s = nolds.sampen(data_s[i, :])  # Sample entropy
            d = nolds.dfa(data_s[i, :])  # Detrended fluctuation analysis

            Lyaps.append(l)
            Hurst.append(h)
            Entropy.append(s)
            dfa.append(d)

        features = [Lyaps, Hurst, Entropy, dfa]
        return np.array(features).T
    
    def plot_chaotic_features(self):
        features = self.chaotic_feature_extractor()
        labels = ['Lyapunov Exponent', 'Hurst Exponent', 'Sample Entropy', 'DFA']
        colors = ['black', 'red', 'blue', 'green']

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.ravel()

        for i in range(4):
            axs[i].plot(features[:, i],'o',color=colors[i], label=labels[i])
            axs[i].axvline(x=160,color='orange')
            axs[i].set_xlabel('Data Vector')
            axs[i].set_ylabel(labels[i])
            axs[i].legend()

        plt.tight_layout()
        plt.show()
    
    def plot_2d_features(self):
        features = self.chaotic_feature_extractor()
        labels = ['Lyapunov Exponent', 'Hurst Exponent', 'Sample Entropy', 'DFA']
        colors = ['black', 'red', 'blue', 'green']

        pairs = list(itertools.combinations(range(len(labels)), 2))  # Generate all pairs of feature indices
        pair_labels = [(labels[x], labels[y]) for x, y in pairs]

        fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjust the number of subplots accordingly
        axs = axs.ravel()

        for i, (x, y) in enumerate(pairs):
            axs[i].scatter(features[:,x],features[:,y],label='Non Epileptic',alpha=0.7, edgecolors='black')
            axs[i].scatter(features[:,x][200:300],features[:,y][200:300],label='Epileptic',c='r', alpha=0.7, edgecolors='black')
            axs[i].set_xlabel(pair_labels[i][0])
            axs[i].set_ylabel(pair_labels[i][1])
            axs[i].set_title(f'{pair_labels[i][0]} vs {pair_labels[i][1]}')

        plt.tight_layout()
        plt.show()

# Example usage:
# classifier = EpilepsyClassifier()
# classifier.load_data()
# classifier.plot_chaotic_features()
# classifier.plot_2d_features()
{% endhighlight %}




Background image credits : <a href="https://commons.wikimedia.org/wiki/File:EEG_Recording_Cap.jpg">Chris Hope</a>, <a href="https://creativecommons.org/licenses/by/2.0">CC BY 2.0</a>, via Wikimedia Commons

