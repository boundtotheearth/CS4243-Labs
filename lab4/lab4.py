import cv2
import numpy as np
import math
from numpy.core.fromnumeric import shape
from sklearn.cluster import KMeans
from itertools import combinations

### Part 1

def detect_points(img, min_distance, rou, pt_num, patch_size, tau_rou, gamma_rou):
    """
    Patchwise Shi-Tomasi point extraction.

    Hints:
    (1) You may find the function cv2.goodFeaturesToTrack helpful. The initial default parameter setting is given in the notebook.

    Args:
        img: Input RGB image. 
        min_distance: Minimum possible Euclidean distance between the returned corners. A parameter of cv2.goodFeaturesToTrack
        rou: Parameter characterizing the minimal accepted quality of image corners. A parameter of cv2.goodFeaturesToTrack
        pt_num: Maximum number of corners to return. A parameter of cv2.goodFeaturesToTrack
        patch_size: Size of each patch. The image is divided into several patches of shape (patch_size, patch_size). There are ((h / patch_size) * (w / patch_size)) patches in total given a image of (h x w)
        tau_rou: If rou falls below this threshold, stops keypoint detection for that patch
        gamma_rou: Decay rou by a factor of gamma_rou to detect more points.
    Returns:
        pts: Detected points of shape (N, 2), where N is the number of detected points. Each point is saved as the order of (height-corrdinate, width-corrdinate)
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w, c = img.shape

    Np = pt_num * 0.9 # The required number of keypoints for each patch. `pt_num` is used as a parameter, while `Np` is used as a stopping criterion.

    # YOUR CODE HERE
    Np = round(Np)
    detected_points = []

    for patch_row in range(0, h, patch_size):
        for patch_col in range(0, w, patch_size):
            scaled_rou = rou
            patch = img_gray[patch_row:patch_row + patch_size, patch_col:patch_col + patch_size]
            patch_keypoints = []
            while(scaled_rou > tau_rou):
                patch_keypoints = cv2.goodFeaturesToTrack(patch, Np, scaled_rou, min_distance)
                if(patch_keypoints is not None and len(patch_keypoints) >= Np):
                    break
                scaled_rou *= gamma_rou

            for i in patch_keypoints:
                x, y = i.ravel()
                detected_points.append([y + patch_row, x + patch_col])

    pts = np.array(detected_points).astype(int)

    # END

    return pts

def extract_point_features(img, pts, window_patch):
    """
    Extract patch feature for each point.

    The patch feature for a point is defined as the patch extracted with this point as the center.

    Note that the returned pts is a subset of the input pts. 
    We discard some of the points as they are close to the boundary of the image and we cannot extract a full patch.

    Args:
        img: Input RGB image.
        pts: Detected point corners from detect_points().
        window_patch: The window size of patch cropped around the point. The final patch is of size (5 + 1 + 5, 5 + 1 + 5) = (11, 11). The center is the given point.
                      For example, suppose an image is of size (300, 400). The point is located at (50, 60). The window size is 5. 
                      Then, we use the cropped patch, i.e., img[50-5:50+5+1, 60-5:60+5+1], as the feature for that point. The patch size is (11, 11), so the dimension is 11x11=121.
    Returns:
        pts: A subset of the input points. We can extract a full patch for each of these points.
        features: Patch features of the points of the shape (N, (window_patch*2 + 1)^2), where N is the number of points
    """


    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float)
    h, w, c = img.shape

    # YOUR CODE HERE
    features = []
    accepted_pts = []
    for pt in pts:
        if(pt[0] - window_patch < 0 or pt[0] + window_patch + 1 >= h or pt[1] - window_patch < 0 or pt[1] + window_patch + 1 >= w):
            continue
        
        accepted_pts.append(pt)
        patch = img_gray[pt[0] - window_patch:pt[0] + window_patch + 1, pt[1] - window_patch:pt[1] + window_patch + 1]
        # normalized_patch = (patch - patch.mean()) / patch.std()
        # patch_features = normalized_patch.ravel()
        patch_features = patch.ravel()
        normalized_patch_features = (patch_features - patch_features.mean()) / patch_features.std()
        features.append(normalized_patch_features)
    
    features = np.array(features)
    # normalised_features = (features - features.mean(axis=0, keepdims=True)) / features.std(axis=0, keepdims=True)
    normalised_features = features
    pts = np.array(accepted_pts)
    # End

    return pts, normalised_features

def mean_shift_clustering(features, bandwidth):
    """
    Mean-Shift Clustering.

    There are various ways of implementing mean-shift clustering. 
    The provided default bandwidth value may not be optimal to your implementation.
    Please fine-tune the bandwidth so that it can give the best result.

    Args:
        img: Input RGB image.
        bandwidth: If the distance between a point and a clustering mean is below bandwidth, this point probably belongs to this cluster.
    Returns:
        clustering: A dictionary, which contains three keys as follows:
                    1. cluster_centers_: a numpy ndarrary of shape [N_c, 2]. Each row is the center of that cluster.
                    2. labels_:  a numpy nadarray of shape [N,], where N is the number of features. 
                                 labels_[i] denotes the label of the i-th feature. The label is between [0, N_c - 1]
                    3. bandwidth: bandwith value
    """
    # YOUR CODE HERE
    accept_radius = bandwidth * (1 / 2)
    threshold = 0.01
    cluster_labels = {}

    clusters = []
    for feature in features:
        clusters.append({'centroid': feature, 'cluster_points': set([tuple(feature)])})
        cluster_labels[tuple(feature)] = -1

    while True:
        print("Iterate")
        new_clusters = []
        # Shift centroids
        converged = True
        len_temp = len(clusters)
        i = 0
        for cluster in clusters:
            # print(i, "of", len_temp)
            # i += 1
            centroid = cluster['centroid']
            absorbed = False
            for new_cluster in new_clusters:
                if(np.linalg.norm(np.subtract(centroid, new_cluster['centroid'])) < accept_radius):
                    absorbed = True
                    new_cluster['cluster_points'] = new_cluster['cluster_points'].union(cluster['cluster_points'])
                    break
            if(absorbed):
                continue

            feature_norms = np.linalg.norm(np.subtract(features, centroid), axis=1)
            in_bandwidth_index = np.nonzero(feature_norms < bandwidth)
            in_bandwidth = features[in_bandwidth_index]
            new_centroid = np.average(in_bandwidth, axis=0)

            if(np.linalg.norm(np.subtract(centroid, new_centroid)) > threshold):
                converged = False
            
            new_clusters.append({'centroid': new_centroid, 'cluster_points': cluster['cluster_points']})
            
        clusters = new_clusters
        if(converged):
            break
    
    cluster_centers = []
    for id, cluster in enumerate(clusters):
        for cluster_point in cluster['cluster_points']:
            cluster_labels[cluster_point] = id
        cluster_centers.append(cluster['centroid'])
    
    cluster_labels = np.array(list(cluster_labels.values()))
    cluster_centers = np.array(cluster_centers)

    clusters = {
        'labels_': cluster_labels,
        'cluster_centers_': cluster_centers,
        'bandwidth_': bandwidth
    }
    # END

    return clusters

def cluster(img, pts, features, bandwidth, tau1, tau2, gamma_h):
    """
    Group points with similar appearance, then refine the groups.

    "gamma_h" provides another way of fine-tuning bandwidth to avoid the number of clusters becoming too large.
    Alternatively, you can ignore "gamma_h" and fine-tune bandwidth by yourself.

    Args:
        img: Input RGB image.
        pts: Output from `extract_point_features`.
        features: Patch feature of points. Output from `extract_point_features`.
        bandwidth: Window size of the mean-shift clustering. In pdf, the bandwidth is represented as "h", but we use "bandwidth" to avoid the confusion with the image height
        tau1: Discard clusters with less than tau1 points
        tau2: Perform further clustering for clusters with more than tau2 points using K-means
        gamma_h: To avoid the number of clusters becoming too large, tune the bandwidth by gradually increasing the bandwidth by a factor gamma_h
    Returns:
        clusters: A list of clusters. Each cluster saves the points that belong to this cluster.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float)
    h, w, c = img.shape

    # YOUR CODE HERE
    clusters = {}
    tuned_banedwidth = bandwidth
    for i in range(10000):
        print("Tuning bandwidth", tuned_banedwidth)
        clusters = mean_shift_clustering(features, tuned_banedwidth)
        if(len(clusters) < len(pts) / 3):
            break
        tuned_banedwidth *= gamma_h

    cluster_points = [[] for i in range(max(clusters['labels_']) + 1)] 
    cluster_features = [[] for i in range(max(clusters['labels_']) + 1)]

    for id, label in enumerate(clusters['labels_']):
        cluster_features[label].append(features[id])
        cluster_points[label].append(pts[id])

    print("Processing Clusters")
    processed_clusters = []
    for cluster_id, feature_set in enumerate(cluster_features):
        feature_set_size = len(feature_set)
        if(feature_set_size < tau1):
            print("Too small")
            continue
        if(feature_set_size < tau2):
            print("Just right")
            processed_clusters.append(np.array(cluster_points[cluster_id]))
            continue
        
        print("Too big")
        k = (feature_set_size // tau2) + 1
        kmeans_results = KMeans(n_clusters=k, random_state = 0).fit(feature_set)
        print("KMeans gives", len(kmeans_results.cluster_centers_))
        new_clusters = [[] for center in kmeans_results.cluster_centers_]

        for id, label in enumerate(kmeans_results.labels_):
            new_clusters[label].append(cluster_points[cluster_id][id])
        for new_cluster in new_clusters:
            processed_clusters.append(np.array(new_cluster))
    clusters = processed_clusters

    # END

    return clusters

### Part 2


def get_proposal(pts_cluster, tau_a, X):
    """
    Get the lattice proposal

    Hints:
    (1) As stated in the lab4.pdf, we give priority to points close to each other when we sample a triplet.
        This statement means that we can start from the three closest points and iterate N_a times.
        There is no need to go through every triplet combination.
        For instance, you can iterate over each point. For each point, you choose 2 of the 10 nearest points. The 3 points form a triplet.
        In this case N_a = num_points * 45.

    (2) It is recommended that you reorder the 3 points. 
        Since {a, b, c} are transformed onto {(0, 0), (1, 0), (0, 1)} respectively, the point a is expected to be the vertex opposite the longest side of the triangle formed by these three points

    (3) Another way of refining the choice of triplet is to keep the triplet whose angle (between the edges <a, b> and <a, c>) is within a certain range.
        The range, for instance, is between 20 degrees and 120 degrees.

    (4) You may find `cv2.getAffineTransform` helpful. However, be careful about the HW and WH ordering when you use this function.

    (5) If two triplets yield the same number of inliers, keep the one with closest 3 points.

    Args:
        pts_cluster: Points within the same cluster.
        tau_a: The threshold of the difference between the transformed corrdinate and integer positions.
               For example, if a point is transformed into (1.1, -2.03), the closest integer position is (1, -2), then the distance is sqrt(0.1^2 + 0.03^2) (for Euclidean distance case).
               If it is smaller than "tau_a", we consider this point as inlier.
        X: When we compute the inliers, we only consider X nearest points to the point "a". 
    Returns:
        proposal: A list of inliers. The first 3 inliers are {a, b, c}. 
                  Each inlier is a dictionary, with key of "pt_int" and "pt" representing the integer positions after affine transformation and orignal coordinates.
    """
    # YOU CODE HERE
    # Build a map of the X closest points to each point
    closest_points = {}
    for point in pts_cluster:
        points = []
        for other_point in pts_cluster:
            distance = np.linalg.norm(np.subtract(point, other_point))
            points.append((other_point, distance))
        sorted_distances = sorted(points, key=lambda x: x[1])
        closest_points[tuple(point)] = sorted_distances[:X]
        

    proposal = []
    proposal_triplet_distance = float('inf')
    for point in pts_cluster:
        samples = combinations(closest_points[tuple(point)][1:11], r=2)

        for point1, point2 in samples:
            x = point
            y, y_dist = point1
            z, z_dist = point2
            xy = y_dist
            xz = z_dist
            yz = np.linalg.norm(np.subtract(y, z))
            triplet_distance = xy + xz + yz

            # Reorder to a, b, c
            if(xy > xz and xy > yz):
                a = z
                b = x
                c = y
            elif(xz > xy and xz > yz):
                a = y
                b = x
                c = z
            else:
                a = x
                b = y
                c = z
            
            # Refine based on angle
            ba = b - a
            ca = c - a
            cosine = np.dot(ba, ca) / (np.linalg.norm(ba) * np.linalg.norm(ca))
            # print(cosine)
            angle = np.arccos(max(min(cosine, 1), -1)) # clamp to [-1, 1] due to floating point errors
            # print(angle)
            if(abs(angle) < 20 * (math.pi / 180) or angle > 120 * (math.pi / 180)):
                continue

            M = cv2.getAffineTransform(np.array([a, b, c])[:, [1, 0]].astype(np.float32), np.array([(0, 0), (0, 1), (1, 0)]).astype(np.float32))

            inliers = [
                {'pt_int': np.array([0, 0]), 'pt': a},
                {'pt_int': np.array([1, 0]), 'pt': b},
                {'pt_int': np.array([0, 1]), 'pt': c},
            ]

            closest_X_points = closest_points[tuple(a)]
            for p, distance in closest_X_points:
                if(np.array_equal(p, a) or np.array_equal(p, b) or np.array_equal(p, c)):
                    continue
                transformed_p = np.flip(np.transpose(np.matmul(M, np.transpose(np.append(np.flip(p, axis=0), 1)))), axis=0)
                int_p = np.round(transformed_p).astype(int)
                distance_to_integer_point = np.linalg.norm(int_p - transformed_p)

                if(distance_to_integer_point < tau_a):
                    inliers.append({'pt_int': int_p, 'pt': p})
            if(len(inliers) > len(proposal)):
                proposal = inliers
                proposal_triplet_distance = triplet_distance
            elif(len(inliers) == len(proposal)):
                if(triplet_distance < proposal_triplet_distance):
                    proposal = inliers
                    proposal_triplet_distance = triplet_distance

    # END

    return proposal





def find_texels(img, proposal, texel_size=50):
    """
    Find texels from the given image.

    Hints:
    (1) This function works on RGB image, unlike previous functions such as point detection and clustering that operate on grayscale image.

    (2) You may find `cv2.getPerspectiveTransform` and `cv2.warpPerspective` helpful.
        Please refer to the demo in the notebook for the usage of the 2 functions.
        Be careful about the HW and WH ordering when you use this function.
    
    (3) As stated in the pdf, each texel is defined by 3 or 4 inlier keypoints on the corners.
        If you find this sentence difficult to understand, you can go to check the demo.
        In the demo, a corresponding texel is obtained from 3 points. The 4th point is predicted from the 3 points.


    Args:
        img: Input RGB image
        proposal: Outputs from get_proposal(). Proposal is a list of inliers.
        texel_size: The patch size (U, V) of the patch transformed from the quadrilateral. 
                    In this implementation, U is equal to V. (U = V = texel_size = 50.) The texel is a square.
    Returns:
        texels: A numpy ndarray of the shape (#texels, texel_size, texel_size, #channels).
    """
    # YOUR CODE HERE
    texels = []
    # triplets = combinations(proposal, 3)
    # for triplet in triplets:
    #     corners_src = np.array(list(map(lambda x: list(x['pt']), triplet)))
    #     point_fourth = corners_src[1] + (corners_src[0] - corners_src[1]) + (corners_src[2] - corners_src[1])
    #     # print('The predicted 4th point:', point_fourth)
    #     corners_src = np.concatenate([corners_src, [point_fourth]])
    #     corners_src = np.array(corners_src).astype(np.float32)
    #     corners_dst = np.float32([[ 0,  0],
    #                             [texel_size,  0],
    #                             [texel_size, texel_size],
    #                             [0, texel_size]])
    #     matrix_projective = cv2.getPerspectiveTransform(corners_src[:, [1, 0]], corners_dst) # transpose (h, w), as the input argument of cv2.getPerspectiveTransform is (w, h) ordering
    #     texel = cv2.warpPerspective(img, matrix_projective, (texel_size, texel_size))
    #     texels.append(texel)
    triplets = combinations(proposal, 3)
    for triplet in triplets:
        corners_src = np.array(list(map(lambda x: list(x['pt']), triplet)))
        x = 0
        y = 1
        z = 2
        xy = np.linalg.norm(np.subtract(corners_src[x], corners_src[y]))
        yz = np.linalg.norm(np.subtract(corners_src[y], corners_src[z]))
        xz = np.linalg.norm(np.subtract(corners_src[x], corners_src[z]))

        # Reorder to a, b, c
        if(xy > xz and xy > yz):
            a = z
            b = x
            c = y
        elif(xz > xy and xz > yz):
            a = y
            b = x
            c = z
        else:
            a = x
            b = y
            c = z
        
        corners_int = np.array(list(map(lambda x: list(x['pt_int']), triplet)))
        if(np.array_equal(corners_int[a], corners_int[b]) or np.array_equal(corners_int[a], corners_int[c]) or np.array_equal(corners_int[c], corners_int[b])):
            continue
        # Refine based on angle
        ba = corners_int[b] - corners_int[a]
        ca = corners_int[c] - corners_int[a]
        if(abs(np.linalg.norm(ba) - 1) > 0.1 or abs(np.linalg.norm(ca) - 1) > 0.1):
            continue
        cosine = np.dot(ba, ca) / (np.linalg.norm(ba) * np.linalg.norm(ca))
        angle = np.arccos(max(min(cosine, 1), -1)) # clamp to [-1, 1] due to floating point errors
        if(abs(angle - (math.pi / 2)) > 0.1):
            continue

        d = corners_src[a] + (corners_src[b] - corners_src[a]) + (corners_src[c] - corners_src[a])

        corners_src = [corners_src[a], corners_src[b], corners_src[c], d]
        corners_src = np.array(corners_src).astype(np.float32)
        corners_dst = np.float32([[ 0,  0],
                                [texel_size,  0],
                                [0, texel_size],
                                [texel_size, texel_size]])
        matrix_projective = cv2.getPerspectiveTransform(corners_src[:, [1, 0]], corners_dst) # transpose (h, w), as the input argument of cv2.getPerspectiveTransform is (w, h) ordering
        texel = cv2.warpPerspective(img, matrix_projective, (texel_size, texel_size))
        texels.append(texel)

    texels = np.array(texels)
    # END
    return texels

def score_proposal(texels, a_score_count_min=3):
    """
    Calcualte A-Score.

    Hints:
    (1) Each channel is normalized separately.
        The A-score for a RGB texel is the average of 3 A-scores of each channel.

    (2) You can return 1000 (in our example) to denote a invalid A-score.
        An invalid A-score is usually results from clusters with less than "a_score_count_min" texels.

    Args:
        texels: A numpy ndarray of the shape (#texels, window, window, #channels).
        a_score_count_min: Minimal number of texels we need to calculate the A-score.
    Returns:
        a_score: A-score calculated from the texels. If there are no sufficient texels, return 1000.    
    """
    if(len(texels) < a_score_count_min):
        return 1000

    K, U, V, C = texels.shape

    # YOUR CODE HERE



    normalized_texels = []
    for texel in texels:
        normalised = (texel - texel.mean(axis=(0,1,2), keepdims=True)) / texel.std(axis=(0,1,2), keepdims=True)
        normalized_texels.append(normalised)

    normalized_texels = np.array(normalized_texels)

    a_score = 0
    for channel in range(C):
        # print(texels[:, :, :, channel].shape)
        combined_texels_channel = normalized_texels[:, :, :, channel]
        # print(combined_texels_channel)
        channel_a_score = 0
        for i in range(U):
            for j in range(V):
                elements = combined_texels_channel[:, i, j]
                channel_a_score += np.std(elements)
        channel_a_score = channel_a_score / (U * V * math.sqrt(K))
        a_score += channel_a_score
    
    a_score /= C
    
    # END

    return a_score


### Part 3
# You are free to change the input argument of the functions in Part 3.
# GIVEN
def non_max_suppression(response, suppress_range, threshold=None):
    """
    Non-maximum Suppression for translation symmetry detection

    The general approach for non-maximum suppression is as follows:
        1. Perform thresholding on the input response map. Set the points whose values are less than the threshold as 0.
        2. Find the largest response value in the current response map
        3. Set all points in a certain range around this largest point to 0. 
        4. Save the current largest point
        5. Repeat the step from 2 to 4 until all points are set as 0. 
        6. Return the saved points are the local maximum.

    Args:
        response: numpy.ndarray, output from the normalized cross correlation
        suppress_range: a tuple of two ints (H_range, W_range). The points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    Returns:
        threshold: int, points with value less than the threshold are set to 0
    """
    H, W = response.shape[:2]
    H_range, W_range = suppress_range
    res = np.copy(response)

    if threshold is not None:
        res[res<threshold] = 0

    idx_max = res.reshape(-1).argmax()
    x, y = idx_max // W, idx_max % W
    point_set = set()
    while res[x, y] != 0:
        point_set.add((x, y))
        res[max(x - H_range, 0): min(x+H_range, H), max(y - W_range, 0):min(y+W_range, W)] = 0
        idx_max = res.reshape(-1).argmax()
        x, y = idx_max // W, idx_max % W
    for x, y in point_set:
        res[x, y] = response[x, y]
    return res

def template_match(img, proposal, threshold):
    """
    Perform template matching on the original input image.

    Hints:
    (1) You may find cv2.copyMakeBorder and cv2.matchTemplate helpful. The cv2.copyMakeBorder is used for padding.
        Alternatively, you can use your implementation in Lab 1 for template matching.

    (2) For non-maximum suppression, you can either use the one you implemented for lab 1 or the code given above.

    Returns:
        response: A sparse response map from non-maximum suppression. 
    """
    # YOUR CODE HERE

    a = proposal[0]['pt']
    b = proposal[1]['pt']
    c = proposal[2]['pt']
    d = a + (b - a) + (c - a)

    min_h = min([a[0], b[0], c[0], d[0]])
    max_h = max([a[0], b[0], c[0], d[0]])
    min_w = min([a[1], b[1], c[1], d[1]])
    max_w = max([a[1], b[1], c[1], d[1]])
    template = img[min_h:max_h, min_w:max_w]
    border_h = (template.shape[0] // 2) + 1
    border_w = (template.shape[1] // 2) + 1
    print(a, b, c, d)
    padded_img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_REFLECT)
    match_result = cv2.matchTemplate(padded_img, template, method=cv2.TM_CCORR_NORMED)
    response = non_max_suppression(match_result, (border_h, border_w), threshold=threshold)
    
    # END
    return response

def maxima2grid(img, proposal, response):
    """
    Estimate 4 lattice points from each local maxima.

    Hints:
    (1) We can transfer the 4 offsets between the center of the original template and 4 lattice unit points to new detected centers.

    Args:
        response: The response map from `template_match()`.

    Returns:
        points_grid: an numpy ndarray of shape (N, 2), where N is the number of grid points.
    
    """
    # YOUR CODE HERE

    a = proposal[0]['pt']
    b = proposal[1]['pt']
    c = proposal[2]['pt']
    d = a + (b - a) + (c - a)

    min_h = min([a[0], b[0], c[0], d[0]])
    max_h = max([a[0], b[0], c[0], d[0]])
    min_w = min([a[1], b[1], c[1], d[1]])
    max_w = max([a[1], b[1], c[1], d[1]])

    center = [(max_h + min_h) / 2, (max_w + min_w) / 2]

    a_displacement = a - center
    b_displacement = b - center
    c_displacement = c - center
    d_displacement = d - center

    maxima = np.where(response > 0)
    maxima = np.array(list(zip(maxima[0], maxima[1])))

    points_grid = []
    for point in maxima:
        points_grid.append(point + a_displacement)
        points_grid.append(point + b_displacement)
        points_grid.append(point + c_displacement)
        points_grid.append(point + d_displacement)

    # END

    return np.array(points_grid)

def refine_grid(img, proposal, points_grid):
    """
    Refine the detected grid points.

    Args:
        points_grid: The output from the `maxima2grid()`.

    Returns:
        points: A numpy ndarray of shape (N, 2), where N is the number of refined grid points.
    """
    # YOUR CODE HERE

    a = proposal[0]['pt']
    b = proposal[1]['pt']
    c = proposal[2]['pt']
    d = a + (b - a) + (c - a)

    min_h = min([a[0], b[0], c[0], d[0]])
    max_h = max([a[0], b[0], c[0], d[0]])
    min_w = min([a[1], b[1], c[1], d[1]])
    max_w = max([a[1], b[1], c[1], d[1]])

    basis_size = np.sqrt((max_h - min_h) ** 2 + (max_w - min_w) ** 2)
    bandwidth = basis_size / 5

    clustering = mean_shift_clustering(points_grid, bandwidth=bandwidth)
    points = clustering['cluster_centers_']
    points = np.array(points).astype(int)

    # END

    return points

def grid2latticeunit(img, proposal, points):
    """
    Convert each lattice grid point into integer lattice grid.

    Hints:
    (1) Since it is difficult to know whether two points should be connected, one way is to map each point into an integer position.
        The integer position should maintain the spatial relationship of these points.
        For instance, if we have three points x1=(50, 50), x2=(70, 50) and x3=(70, 70), we can map them (4, 5), (5, 5) and (5, 6).
        As the distances between (4, 5) and (5, 5), (5, 5) and (5, 6) are both 1, we know that (x1, x2) and (x2, x3) form two edges.
    
    (2) You can use affine transformation to build the mapping above, but do not perform global affine transformation.

    (3) The mapping in the hints above are merely to know whether two points should be connected. 
        If you have your own method for finding the relationship, feel free to implement your owns and ignore the hints above.


    Returns:
        edges: A list of edges in the lattice structure. Each edge is defined by two points. The point coordinate is in the image coordinate.
    """

    # YOUR CODE HERE
    # print("Proposal: ", proposal)
    # print("Points: ", points)
    edges = []

    # a = proposal[0]['pt']
    # b = proposal[1]['pt']
    # c = proposal[2]['pt']

    # M = cv2.getAffineTransform(np.array([a, b, c])[:, [1, 0]].astype(np.float32), np.array([(0, 0), (0, 1), (1, 0)]).astype(np.float32))

    # transformed_points = []
    # for point in points:
    #     transformed_point = np.flip(np.transpose(np.matmul(M, np.transpose(np.append(np.flip(point, axis=0), 1)))), axis=0)
    #     transformed_points.append(transformed_point)
    
    # # print(np.array(transformed_point))
    
    # for i, point in enumerate(transformed_points):
    #     for j, otherPoint in enumerate(transformed_points):
    #         if(np.array_equal(point, otherPoint)):
    #             continue
    #         if(np.linalg.norm(np.subtract(point, otherPoint)) <= 1.2):
    #             edges.append([points[i], points[j]])

    for point in points:
        # Get nearby points
        point_norms = np.linalg.norm(np.subtract(points, point), axis=1)
        nearby_index = np.nonzero(point_norms < 100)
        nearby_points = points[nearby_index]
            
        local_proposal = get_proposal(nearby_points, 0.2, 10)
        for proposal_a in local_proposal:
            original_a = proposal_a['pt']
            transformed_a = proposal_a['pt_int']
            for proposal_b in local_proposal:
                original_b = proposal_b['pt']
                transformed_b = proposal_b['pt_int']
                if(np.array_equal(transformed_a, transformed_b)):
                    continue
                if(abs(np.linalg.norm(np.subtract(transformed_a, transformed_b)) - 1) < 0.1):
                    edges.append([original_a, original_b])

    edges = np.array(edges)

    # END

    return edges





