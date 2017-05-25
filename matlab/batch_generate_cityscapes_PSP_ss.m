class = 'unified';
set = 'val';

keys = [13, 11, 12, 17, 18, 14, 15, 16];

input_list_file = strcat('./cityscapes/splits/', set,'list.txt');
output_dir = strcat('./cityscapes/unified/ssMaskFinePSP/', set);
input_dir = './PSPNet/evaluation/mc_result/cityscapes/test/gray';


fid = fopen(input_list_file);
input_file = fgetl(fid);
processed = 0;
while ischar(input_file)
    id = regexpi(input_file, '[a-z]+_\d\d\d\d\d\d_\d\d\d\d\d\d', 'match');
    id = id{1};
    city = regexpi(id, '^[a-z]+', 'match');
    city = city{1};
    output_file = fullfile(output_dir, city, strcat(id, '_', class, '_ss.mat'));
    %output_file = fullfile(output_dir, city, strcat(id, '_car_semantic_CF.png'));
    [output_file_dir, ~, ~] = fileparts(output_file);
    if ~exist(output_file_dir, 'dir')
        mkdir(output_file_dir);
    end
    
    raw_mask = imread(fullfile(input_dir, [id, '.png']));
    
    [height, width] = size(raw_mask);
    
    mask = uint8(zeros(height/2, width/2, 8));
    
    for i=1:length(keys)
        raw_mask_downsampled = uint8(raw_mask == keys(i));
        raw_mask_downsampled = downsample(downsample(raw_mask_downsampled,2)', 2)';
        mask(:,:,i) = uint8(raw_mask_downsampled);
    end
    
    save(output_file, 'mask');
    
    input_file = fgetl(fid);
    processed = processed + 1;
    if mod(processed, 100) == 0
        disp(sprintf('Processed %d segmentation masks', processed));
    end

end
fclose(fid);
