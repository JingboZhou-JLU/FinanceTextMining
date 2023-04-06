from sklearn.mixture import GaussianMixture

class cluster_GMM():
    def __init__(self,embed,n_clusters,text_num):
        self.embed = embed.detach().numpy()
        n_samples=self.embed.shape[0]
        if(n_samples<n_clusters):
            self.gmm = GaussianMixture(n_components=n_samples).fit(self.embed)
        else:
            self.gmm = GaussianMixture(n_components=n_clusters).fit(self.embed)
        self.l=text_num
    def get_result(self):
        return self.gmm.predict(self.embed)[:self.l]