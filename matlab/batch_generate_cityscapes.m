class = 'unified';

sets = {'val', 'train'};

for j = 1:2
set = sets{j};

keys = [26000, 26999;
    24000, 24999,
    25000, 25999,
    32000, 32999,
    33000, 33999,
    27000, 27999,
    28000, 28999,
    31000, 31999];

input_list_file = strcat('./cityscapes/splits/',set,'list.txt');
input_folder = strcat('./cityscapes/gtFine/', set);
output_file_path = strcat('./cityscapes/unified/iGTFull/', set);

boundaries = [0,1,2,3,4,5,7,9,12,15,19,24,30,37,45,54,Inf];

fid = fopen(input_list_file);
input_file = fgetl(fid);
processed = 0;

while ischar(input_file)
    id = regexpi(input_file, '[a-z]+_\d\d\d\d\d\d_\d\d\d\d\d\d', 'match');
    id = id{1};
    city = regexpi(id, '^[a-z]+', 'match');
    city = city{1};
    output_file = fullfile(output_file_path, city, strcat(id, strcat('_',class,'_GT.mat')));
    [output_file_dir, ~, ~] = fileparts(output_file);
    if ~exist(output_file_dir, 'dir')
        mkdir(output_file_dir);
    end
    
     generate_GT_cityscapes_unified(fullfile(input_folder, strcat(input_file, '_gtFine_instanceIds.png')), ...
         output_file, false, keys, 1, boundaries);

    input_file = fgetl(fid);
    processed = processed + 1;
    if mod(processed, 50) == 0
        disp(sprintf('Processed %d direction files', processed));
    end
end
fclose(fid);

end
