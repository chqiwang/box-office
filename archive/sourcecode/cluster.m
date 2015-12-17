
feature_path = 'sift_feature';
feature = dir(feature_path);

Des = [];
feature_num = [];
num_clusters = 50;      

fprintf('Loading features...\n');
for i = 1 : length(feature)
    if ~strcmp(feature(i).name,'.') && ~strcmp(feature(i).name,'..')
        sift_feature_path = fullfile(feature_path, feature(i).name);
        load(sift_feature_path);
        Des = [Des, des];                      
        feature_num = [feature_num, size(des, 2)];
    end
end
save('sift_feature.mat', 'Des', 'feature_num');

fprintf('Clustering... \n');

[cluster_centers, assignments] = vl_ikmeans(Des, num_clusters);  
Des = double(Des);
[cluster_centers, assignments] = vl_kmeans(Des, num_clusters);    

save('Codebook.mat', 'cluster_centers', 'feature_num');
fprintf('Done! \n');
