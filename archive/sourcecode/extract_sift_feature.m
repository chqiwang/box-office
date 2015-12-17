
run('vlfeat-0.9.14\toolbox\vl_setup');
image_data_path = 'imgs';   
sift_feature_path = 'sift_feature';  
image = dir(image_data_path);
for i = 1 : length(image)
    if ~strcmp(image(i).name, '.') && ~strcmp(image(i).name, '..')
        disp(['Processing ', image(i).name, '...']);
        im = imread(fullfile(image_data_path, image(i).name));
        if size(im, 3) > 1
            im = rgb2gray(im);
        end
        I = single(im);
        [F, des] = vl_sift(I);
        [pstr, name, ext] = fileparts(image(i).name);
        sift_feature_name = [name, '.mat'];
        save(fullfile(sift_feature_path, sift_feature_name), 'F', 'des');
    end
end
fprintf('Done!\n');
