function generate_GT_cityscapes_unified(input_annotation, gt_output, figures, keys, downsampling, depth_bins)
% generates 2 channel unitvec gt
    annotation = imread(input_annotation);
    
    if downsampling > 1
       annotation = downsample(downsample(annotation,downsampling)', downsampling)'; 
    end
    
    height = size(annotation,1);
    width = size(annotation,2);
    depth_map = zeros(size(annotation));
    dir_map = repmat(depth_map, 1, 1, 2);
    weight_map = depth_map;
    ss_map = depth_map;
    edge_map = depth_map;
    se = strel('disk', 1, 8);
    se3 = strel('disk', 3, 8);
    
    %annotation(annotation < classId | annotation > (classId + 1000)) = 0;
    annotation(~((annotation >= keys(1,1) & annotation <= keys(1,2)) | ...
        (annotation >= keys(2,1) & annotation <= keys(2,2)) | ...
        (annotation >= keys(3,1) & annotation <= keys(3,2)) | ...
        (annotation >= keys(4,1) & annotation <= keys(4,2)) | ...
        (annotation >= keys(5,1) & annotation <= keys(5,2)) | ...
        (annotation >= keys(6,1) & annotation <= keys(6,2)) | ...
        (annotation >= keys(7,1) & annotation <= keys(7,2)) | ...
        (annotation >= keys(8,1) & annotation <= keys(8,2)))) = 0;
    
    ss_map = annotation;
    ss_map(ss_map > 1) = 1;
    
    ids = unique(annotation);
    
    for i = 2:length(ids)
       annotation_i = annotation;
       annotation_i(annotation_i~=ids(i)) = 0;
       annotation_i(annotation_i>0) = 1;
       
       if sum(sum(annotation_i)) < 100
           continue;
       end
       
       depth_i = bwdist(1-annotation_i);
       depth_map = depth_map + depth_i;
       
       dir_i = zeros(size(dir_map));
       
       [dir_i(:,:,1), dir_i(:,:,2)] = imgradientxy(depth_i);
       
       dir_i = dir_i / 8;
       
       dir_map = dir_map + dir_i;
       
       weight_map(annotation_i==1) = 200 / sqrt(sum(sum(annotation_i)));
    end
    
    edges = 1-double(edge(annotation));
    dir_map = dir_map .* repmat(edges, 1, 1, 2);
    %depth_map = depth_map .* edges;
    
    for i=1:length(depth_bins)-1
       depth_map(depth_map > depth_bins(i) & depth_map <= depth_bins(i+1)) = i-1;
    end    
    
    for i=1:size(keys,1)
       annotation_i = double(annotation);
       annotation_i(annotation_i < keys(i,1) | annotation_i > keys(i,2)) = 0;
       annotation_i = annotation_i - (keys(i,1)) + double(~annotation_i) * (keys(i,1));
       edge_map_i = edge(annotation_i, 0.00001);
       
       annotation_i_inv = ~(imdilate(~annotation_i, se));
       
       edge_map_i = edge_map_i .* annotation_i_inv;
       edge_map_i = imdilate(edge_map_i, se);
       edge_map = max(edge_map, edge_map_i);
    end
    
    dir_map = single(dir_map);
    depth_map = uint8(depth_map);
    weight_map = single(weight_map);
    ss_map = uint8(ss_map);
    edge_map = uint8(edge_map);
    
    save(gt_output, 'depth_map', 'dir_map', 'weight_map', 'edge_map');
    if figures
        figure(1);
        imagesc(depth_map);
        figure(2);
        imagesc(dir_map(:,:,1));
        figure(3);
        imagesc(dir_map(:,:,2));
        figure(3);
        imagesc(weight_map);
        figure(4);
        imagesc(ss_map);
    end
end

