import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the 
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    centers = []
    centers.append(p)

    for i in range(n_cluster-1):
        
        distance = []
        for j in range(len(x)):
            d = float('inf')
            candidate_point = x[j]
            centriod_size = len(centers)
            for k in range(len(centers)):
                temp_distance = np.sum((x[centers[k]]-candidate_point)**2)
                if(temp_distance < d):
                    d = temp_distance
            distance.append(d)

        sum_of_distances = sum(distance)
        #Normalizing distance metric to find cumulative probability
        for dist in range(len(distance)):
            distance[dist] = distance[dist]/sum_of_distances
        
        
        r = generator.rand()
        cumulative_probability = 0
        
        for index in range(len(distance)):
            cumulative_probability += distance[index]
            if(cumulative_probability>r):
                next_mu = index
                break
        
        centers.append(next_mu)
    #first_center_index = generator.randint(0, n)
    #p = generator.randint(0, n)
    #centers_point = []
    #centers = []
    #centers_point.append(x[p])
    #centers.append(p)

    #for k in range(n_cluster - 1):
    #    distance_to_closest_centroid = []
    #    for i in range(len(x)):
    #        min_distance = float("inf")
    #        for j in range(len(centers_point)):
    #            center = centers_point[j]
    #            distance = np.sum((x[i] - center)**2)
    #            if distance < min_distance:
    #                min_distance = distance
    #        distance_to_closest_centroid.append(min_distance)
    #    sum_distances = sum(distance_to_closest_centroid)
    #    for m in range(len(distance_to_closest_centroid)):
    #        distance_to_closest_centroid[m] = distance_to_closest_centroid[m] / sum_distances
        
    #    r = generator.rand()
    #    cumulative_probability = 0
        
    #    for index in range(len(distance_to_closest_centroid)):
    #        cumulative_probability += distance_to_closest_centroid[index]
    #        if(cumulative_probability>r):
    #            next_mu = index
    #            break
    #    #index = get_center_by_custom_logic(generator, distance_to_closest_centroid)
    #    centers.append(next_mu)
    #    centers_point.append(x[next_mu])
          
    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        centroids = np.zeros([self.n_cluster, D])
        for i in range(self.n_cluster):
            centroids[i] = x[self.centers[i]]

        cluster_indice_map = np.zeros(N)
        
        kmeans_objective = 0.0
        for i in range(self.n_cluster):
            kmeans_objective += np.sum((x[cluster_indice_map == i] - centroids[i]) ** 2)
            

        kmeans_objective = kmeans_objective/N

        iter_count = 0
        while iter_count < self.max_iter:
            iter_count += 1
            cluster_indice_map = np.argmin(np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2), axis=0)
            
            kmeans_objective_new = 0.0
            for i in range(self.n_cluster):
                kmeans_objective_new += np.sum((x[cluster_indice_map == i] - centroids[i]) ** 2)
                
            
            kmeans_objective_new = kmeans_objective_new/N

            if abs(kmeans_objective - kmeans_objective_new) <= self.e:
                break
            
            kmeans_objective = kmeans_objective_new
            
            centroids_new = np.array([np.mean(x[cluster_indice_map == i], axis=0) for i in range(self.n_cluster)])
            
            index = np.where(np.isnan(centroids_new))
            
            centroids_new[index] = centroids[index]
            centroids = centroids_new
        
        centroids = np.asarray(centroids)
        return centroids, cluster_indice_map, iter_count


        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        KmeansObj =  KMeans(self.n_cluster, self.max_iter,self.e, self.generator)
        centroids, predicted_labels, max_iter = KmeansObj.fit(x, centroid_func = centroid_func)
        
        assigned_labels = [[] for i in range(self.n_cluster)]
        for i in range(N):
            assigned_labels[predicted_labels[i]].append(y[i])

        centroid_labels = np.zeros([self.n_cluster])
        
        for i in range(self.n_cluster):
            unique,counts = np.unique(assigned_labels[i], return_counts=True)
            index = np.argmax(counts)
            centroid_labels[i] = unique[index]

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        predicted_labels = np.zeros(N)
        for i in range(len(x)):
            min_distance = float('inf')
            for j in range(len(self.centroids)):
                temp_distance = np.sum((x[i]-self.centroids[j])**2)
                if(temp_distance < min_distance):
                    min_distance = temp_distance
                    predicted_labels[i] = self.centroid_labels[j]
        
        return predicted_labels




def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''
    
    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    reshaped_image = np.reshape(image, [image.shape[0]*image.shape[1], image.shape[2]])
    #Using similar logic used in KMeans to find the closest centroid index
    nearest_centroid = np.argmin(np.sum(((reshaped_image - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2), axis=0)
    transformed_image = code_vectors[nearest_centroid].reshape(image.shape)
        
    return transformed_image
     

