
num_clusters = 50;
load('Codebook.mat');
load('sift_feature.mat');
cluster_centers = double(cluster_centers);
Img_num = length(feature_num);
k = 0;
image_words = zeros(num_clusters, Img_num);
for i = 1 : Img_num
    Img_fea = Des(:, (k + 1) : (k + feature_num(i)));
    d = EuclideanDistance(double(cluster_centers'), double(Img_fea'));
    [minz, index] = min(d, [], 1); 
    image_words(:, i)= hist(index, (1 : 50));
end

image_words = image_words ./ repmat(sum(abs(image_words)), [size(image_words, 1), 1]);
save('image_words.mat', 'image_words');
csvwrite('image_words.csv', image_words);

