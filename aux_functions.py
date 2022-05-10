'''
@author: Bryan T. Kim
@title: helper funcions for hierarchical clustering
@content:

    def eda_subcorpus(sample, subset_cat, label, time_period)
    def reduce_dim(cos_sim, rand_state)
    def tokenize_and_stem(text)
    def tfidf_vectorizer(_raw_text)
    class TopToolbar(mpld3.plugins.PluginBase)
    def _plot_freq(df, feature, by, agg=False)
    def _plot_dendrogram(cos_sim, target, zoom_in=True, threshold=0, save_pic=False)
    def custom_css()
    def _plot_clusters(df, save=False)
'''

def eda_subcorpus(sample, subset_cat, label, year, time_period):
    sub_corpus = sample[sample[label].str.contains('|'.join(subset_cat))][['year','month','day','section_clean','title_clean', 'article_clean']]
    print('Category - ' + ' '.join(subset_cat).upper() + ' : ', len(sub_corpus))
    print('Subcorpus Dim :', sub_corpus.shape)
    print('Counts by ' + time_period + ' for year ' + year)
    print(sub_corpus.groupby(time_period)[label].count())

    return sub_corpus

from sklearn.cluster import AgglomerativeClustering
def clustering(tfidf_matrix, _n_clusters):
    cluster = AgglomerativeClustering(n_clusters=_n_clusters, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(tfidf_matrix.toarray())
    return cluster.labels_

from sklearn.manifold import MDS
def reduce_dim(cos_sim, rand_state):
    ## Dimension reduced to 2D for visualization - xs,ys
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=rand_state)
    pos = mds.fit_transform(cos_sim)
    return pos[:, 0], pos[:, 1]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')
import numpy as np
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]    
    stems = [stemmer.stem(t) for t in tokens if t not in stopwords]
    return stems

def tfidf_vectorizer(_raw_text, max_df=.5, min_df=10, gram=3, max_features=20*10000, n_show=20):
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features,
                                   tokenizer=tokenize_and_stem, ngram_range=(1,gram))
    tfidf_matrix = tfidf_vectorizer.fit_transform(_raw_text)
    print(tfidf_matrix.shape)

    terms = tfidf_vectorizer.get_feature_names_out()
    print('Feature names up to ' + str(n_show) + ' : ', terms[:n_show])

    tfidf_matrix = tfidf_matrix.astype(np.uint8)
    cos_sim = cosine_similarity(tfidf_matrix)

    print('cosine_similarity dim : ', cos_sim.shape)
    print('cosine_similarity matrix : ', cos_sim)

    return cos_sim, tfidf_matrix

import  mpld3
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

import matplotlib.pyplot as plt
def _plot_freq(df, _top_n=10, fw=16, fh=12, _nrows=2, _ncols=5):
    '''
    Plot top n sections over time in a given year set
    '''
    fig, axes = plt.subplots(nrows=_nrows, ncols=_ncols, sharex=False, sharey=True, figsize=(fw,fh))
    axes_list = [item for sublist in axes for item in sublist] 

    ordered_section = df.groupby('section_clean')['title_clean'].count().nlargest(_top_n).index

    for section in ordered_section:
        grouped = df[df['section_clean'] == section].groupby('month')['title_clean'].count()

        ax = axes_list.pop(0)
        grouped.plot(x='month', label=section, ax=ax, linestyle='None', marker='o')
        ax.set_title(section)
        ax.tick_params(
            which='both',
            bottom='off',
            left='off',
            right='off',
            top='off'
        )
        ax.grid(linewidth=0.5)
        ax.set_xlim((.5, 12.5))
    #     ax.set_xlabel('month')
        ax.set_xticks(range(1, 13))
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes_list:
        ax.remove()

    plt.suptitle('Top ' + str(_top_n) + ' sections by month')
    plt.tight_layout(pad=3)
    plt.show()

from scipy.cluster.hierarchy import ward, dendrogram
def _plot_dendrogram(cos_sim, target, _p=30, _trunc_mode=None, fw=15, fh=10
                    , zoom_in=True, zoom_xlim = 2500, threshold=0, save_pic=False):

    linkage_matrix = ward(cos_sim)

    plt.figure(figsize=(fw, fh))
    dendrogram(linkage_matrix
              ,labels=target
              ,above_threshold_color='y'
              ,p=_p
              ,truncate_mode=_trunc_mode
               )
    if zoom_in:
        plt.xlim(0, zoom_xlim)
        plt.title('Dendrogram - zoomed in up to '+ str(zoom_xlim))
    else:
        plt.title('Dendrogram - All data points')
    
    if threshold > 0:
        plt.axhline(y=threshold, color='r', linestyle='--')
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    if save_pic:
        plt.savefig('hierarchical_clusters_dendrogram.png')
        
    plt.close()

class custom_css:
    #define custom css to format the font and to remove the axis labeling
    css = """
        text.mpld3-text, div.mpld3-tooltip {
          font-family:Arial, Helvetica, sans-serif;
        }

        g.mpld3-xaxis, g.mpld3-yaxis {
        display: none; }

        svg.mpld3-figure {
        margin-left: -200px;}
     """

def _plot_clusters(df_2d, save=False):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.margins(0.05)

    for c, group in df_2d.groupby('segment'):
        points = ax.plot(group.x, group.y, marker='o',label=c, linestyle='', ms=15)
        ax.set_aspect('auto')
        clusters = [i for i in group.label]
        
        #set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], clusters, voffset=10, hoffset=10, css=custom_css.css)
        #connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())    
        
        #set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        
        #set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    ax.legend(numpoints=1)

    # mpld3.display()
    plt.show()

    if save:
        html = mpld3.fig_to_html(fig)
        print(html)

    plt.close(fig)