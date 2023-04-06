from sklearn.cluster import KMeans
class cluster_KM():
    def __init__(self,embed,n_clusters,text_num):
        n_samples=embed.shape[0]
        if(n_samples<n_clusters):
            self.kmeans = KMeans(n_clusters=n_samples, random_state=0).fit(embed.detach().numpy())
        else:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embed.detach().numpy())
        self.l=text_num
    def get_result(self):
        return self.kmeans.labels_[:self.l]