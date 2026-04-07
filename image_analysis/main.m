%% Image Analysis
close all; clc; clear; close all;

% ---------------------- load main config file
config_file = load_json_config("config.json");

% ---------------------- open reference file
reference_file = readtable(config_file.reference_file_folder + "reference.csv",'PreserveVariableNames',true, 'Delimiter', ',');
num_data_points = size(reference_file, 1);
% curr date for output file
new_reference_filepath = config_file.reference_file_folder + "reference_" + string(datetime('now','TimeZone','local','Format','dd-MMM-yyyy_HH-mm-ss')) + ".csv";
% add columns for data
data_cols = {"num_splines", "spline_breaks", "spline_coefs", "spline_control_pts"};
save_image_and_masks = true;
if save_image_and_masks
    data_cols{end+1} = "img_and_mask_path";
end
for c = 1:length(data_cols)
    if ~any(ismember(reference_file.Properties.VariableNames, data_cols{c}))
        reference_file.(data_cols{c})(:) = "";
    end
end

% ---------------------- load stereo camera parameters
stereo_params = load(config_file.stereo_params_path);
stereo_params = stereo_params.stereoParams;

% ---------------------- apriltag calibration on first image
apriltag_flag = true;
if apriltag_flag
    left_first_image_path = strcat( config_file.reference_file_folder, reference_file.left_image_path{1} );
    left_first_image = imread(left_first_image_path);
    [left_tag_id, left_tag_loc, left_tag_pose] = readAprilTag(left_first_image, ...
                                                                config_file.apriltag_config.tag_family, ...
                                                                    stereo_params.CameraParameters1.Intrinsics, ...
                                                                        config_file.apriltag_config.tag_size);
    right_first_image_path = strcat( config_file.reference_file_folder, reference_file.right_image_path{1} );
    right_first_image = imread(right_first_image_path);
    abs_tag_transform = left_tag_pose.A * config_file.apriltag_config.tag_transform;
    % get pixel space points for origin
    origin_left_image_point = world2img(abs_tag_transform(1:3, 4)', rigidtform3d, stereo_params.CameraParameters1.Intrinsics);
    origin_right_image_point = world2img(abs_tag_transform(1:3, 4)', stereo_params.PoseCamera2, stereo_params.CameraParameters2.Intrinsics);
else
    % if not using apriltags, use identity transform as the tube origin (point is closer to the tube base)
    abs_tag_transform = eye(4);
    % define pixel coordinates for origin in each image (used to specify which cluster is the tube in segmentation step)
    origin_left_image_point = [1, 1];   % TODO: update based on actual images
    origin_right_image_point = [1, 1];  % TODO: update based on actual images
    fprintf("Skipping apriltag calibration and using identity transform.\n");
end

% parameters for saving images and mask overlays
ylim_crop = [100, 1500];
batch_size = 10000; % num imgs per folder

% ---------------------- loop through all data points
fprintf("Number of data points: %d\n", num_data_points);
for i = 1:num_data_points
    frame_id = reference_file.frame_id(i);
    % do no skip if blank or NaN
    if ~strcmp(reference_file.num_splines(i), "") && ~isnan(reference_file.num_splines(i))
        fprintf("Skipping for image %d.\n", frame_id);
        continue;
    end
    if mod(i, 100) == 0
        fprintf("Performing image analysis for image %d.\n", frame_id);
    end

    try
        %% load and pick mask for tubes
        % load pair of masks (already segmented by SAM)
        left_mask_path = strcat(config_file.reference_file_folder, reference_file.left_mask_path{i});
        right_mask_path = strcat(config_file.reference_file_folder, reference_file.right_mask_path{i});
        left_mask = imread(left_mask_path);
        right_mask = imread(right_mask_path);
        % threshold image
        thresh = 150;
        left_mask = left_mask > thresh;
        right_mask = right_mask > thresh;
        % pick the cluster for the tube via the origin point from apriltag calibration
        left_mask_clusters = bwlabel(left_mask,4);
        right_mask_clusters = bwlabel(right_mask,4);
        left_tube_cluster = left_mask_clusters(int32(origin_left_image_point(2)), int32(origin_left_image_point(1)));
        right_tube_cluster = right_mask_clusters(int32(origin_right_image_point(2)), int32(origin_right_image_point(1)));
        left_mask = left_mask_clusters == left_tube_cluster;
        right_mask = right_mask_clusters == right_tube_cluster;

        %% extract out centerline
        % Get skeleton of image
        left_centerline_mask = crop_bwskel(left_mask);
        right_centerline_mask = crop_bwskel(right_mask);
        left_centerline = image2pts(left_centerline_mask);
        right_centerline = image2pts(right_centerline_mask);
        left_end_points = image2pts(bwmorph(left_centerline_mask, 'endpoints'));
        right_end_points = image2pts(bwmorph(right_centerline_mask, 'endpoints'));
        % Get sorted array of pxiels
        sorted_left_centerline = order_points_along_line(left_centerline);
        sorted_right_centerline = order_points_along_line(right_centerline);
        
        % Extend centerline to mask ends via polynomial fitting
        ord = 2;
        left_centerline_pts_extended = extend_centerlines_in_mask(left_mask, sorted_left_centerline, ord);
        right_centerline_pts_extended = extend_centerlines_in_mask(right_mask, sorted_right_centerline, ord);
        right_extended_centerline_im = pts2image(right_centerline_pts_extended, size(right_mask), [0, 0]);

        % Reconstruct 3D points based on skeleton stereoimages  % takes 0.06-0.08 for both getWorldPts() calls
        [world_pts, right_img_pts] = get_world_points(stereo_params, left_centerline_pts_extended, right_extended_centerline_im);
            
        % sort world pts from base to tip
        [~, first_ind] = choose_closer_point( abs_tag_transform(1:3, 4)', [world_pts(1, :); world_pts(end, :)] );
        if first_ind ~= 1
            world_pts = flipud(world_pts);
        end

        % replace first point with april tag origin
        world_pts(1, :) = abs_tag_transform(1:3, 4)';
        
        % align points to origin in world frame according to apriltag
        aligned_world_points = align_points_to_origin(world_pts, abs_tag_transform);

        % Fit 3D spline to points
        [spline_coefs, spline_breaks, control_pts] = world_pts2spline(aligned_world_points, config_file.reconstruction_config.num_splines);

        % load values to reference file
        reference_file.num_splines(i) = config_file.reconstruction_config.num_splines;
        reference_file.spline_breaks{i} = jsonencode(spline_breaks);
        reference_file.spline_coefs{i} = jsonencode(spline_coefs);
        reference_file.spline_control_pts{i} = jsonencode(control_pts);

        %% save image overlays for verification
        if save_image_and_masks
            % overlay image
            left_image_path = strcat(config_file.reference_file_folder, reference_file.left_image_path{i});
            right_image_path = strcat(config_file.reference_file_folder, reference_file.right_image_path{i});
            left_image = imread(left_image_path);
            right_image = imread(right_image_path);    
            left_image_overlay = labeloverlay(left_image, left_mask, "Transparency", 0.85, "Colormap", [1 1 0]);
            right_image_overlay = labeloverlay(right_image, right_mask, "Transparency", 0.85, "Colormap", [1 1 0]);
            % use saved spline to get points in 3d
            spline_pts = get_spline_points(config_file.reconstruction_config.num_splines, spline_breaks, spline_coefs);
            spline_pts_h = [spline_pts, ones(size(spline_pts, 1), 1)];   % N x 4
            temp_spline_pts_h = (abs_tag_transform * spline_pts_h')';  % N x 4
            aligned_spline_pts = temp_spline_pts_h(:,1:3);
            % project points into each image
            left_image_points = world2img(aligned_spline_pts, rigidtform3d, stereo_params.CameraParameters1.Intrinsics);
            right_image_points = world2img(aligned_spline_pts, stereo_params.PoseCamera2, stereo_params.CameraParameters2.Intrinsics);
            % process images with points
            left_image_marked = insertMarker(left_image_overlay, left_image_points, 'o', MarkerColor=[0.8 0 0], Size=1);
            right_image_marked = insertMarker(right_image_overlay, right_image_points, 'o', MarkerColor=[0.8 0 0], Size=1);
            left_image_cropped = left_image_marked(ylim_crop(1):ylim_crop(2), 50:1200, :);
            right_image_cropped = right_image_marked(ylim_crop(1):ylim_crop(2), 500:1500, :);
            combined = [left_image_cropped, right_image_cropped];
            % get image filename
            split_path = split(reference_file.left_image_path{i}, "_");
            % split_path2 = split(split_path{1}, "/");
            % batch_id = split_path2{2};
            image_id = split_path{3};   % from cam hardware
            batch_id = num2str( floor((i-1) / batch_size) );
            % create relevant folders
            batch_folder = strcat(config_file.reference_file_folder, '/imgs_and_masks/', batch_id, '/');
            if ~exist(batch_folder, 'dir')
                mkdir(batch_folder);
            end
            outfile = strcat(batch_folder, 'img_and_mask_', image_id, "_", num2str(reference_file.frame_id(i) + 1), '.jpg');
            reference_file.img_and_mask_path{i} = char(outfile);
            imwrite(combined, outfile);
        end

        %% save to reference file
        if mod(i, 10000) == 0
            fprintf("Saving data to reference file at frame %d.\n", frame_id);
            writetable(reference_file,new_reference_filepath,'Delimiter',',','QuoteStrings','all', 'WriteMode','overwrite');
        end
    catch ME
        fprintf("Error processing frame %d: %s\n", frame_id, ME.message);
    end
end

fprintf("Saving data to reference file at FINAL frame %d.\n", frame_id);
writetable(reference_file,new_reference_filepath,'Delimiter',',','QuoteStrings','all', 'WriteMode','overwrite');


%% Functions

function json_data = load_json_config(json_filename)
    json_data = struct([]);
    % Check if file exists
    if isfile(json_filename)
        fprintf("Loading JSON config file.\n");
        % Convert JSON string to MATLAB variables
        json_data = jsondecode(fileread(json_filename));
    else
        fprintf("Config file does not exist at path %s.\n", json_filename);
    end
end

function plot_apriltag(image, tag_id, tag_loc, tag_pose, tag_size, intrinsics)
    for idx = 1:length(tag_id)
        % Insert markers to indicate the locations
        markerRadius = 2;
        numCorners = size(tag_loc,1);
        markerPosition = [tag_loc(:,:,idx),repmat(markerRadius,numCorners,1)];
        image = insertShape(image,"FilledCircle",markerPosition,ShapeColor="red",Opacity=1);
    end
    worldPoints = [0 0 0; tag_size/2 0 0; 0 tag_size/2 0; 0 0 tag_size/2];
    for i = 1:length(tag_pose)
        % Get image coordinates for axes.
        imagePoints = world2img(worldPoints,tag_pose(i), intrinsics);

        % Draw colored axes.
        image = insertShape(image,Line=[imagePoints(1,:) imagePoints(2,:); ...
            imagePoints(1,:) imagePoints(3,:); imagePoints(1,:) imagePoints(4,:)], ...
            ShapeColor=["red","green","blue"],LineWidth=7);

        image = insertText(image,tag_loc(1,:,i),tag_id(i),BoxOpacity=1,FontSize=25);
    end
    imshow(image)
end

function plot_frame(frame)
    % Plots a 3D frame given a 4x4 homogeneous transform
    origin = frame(1:3, 4);
    x_axis = frame(1:3, 1) + origin;
    y_axis = frame(1:3, 2) + origin;
    z_axis = frame(1:3, 3) + origin;
    hold on; grid on;
    plot3([origin(1), x_axis(1)], [origin(2), x_axis(2)], [origin(3), x_axis(3)], '-r', 'LineWidth', 2);
    plot3([origin(1), y_axis(1)], [origin(2), y_axis(2)], [origin(3), y_axis(3)], '-g', 'LineWidth', 2);
    plot3([origin(1), z_axis(1)], [origin(2), z_axis(2)], [origin(3), z_axis(3)], '-b', 'LineWidth', 2);
end

function pts = image2pts(im)
    % returns a list of pts (m, 2) in form of [y, x] or [row, col] that
    % correspond to white pixels in binary image 'im'.
    [r, c] = find(im);
    pts = horzcat(r, c);
end

function im = pts2image(pts, sz, offsets)
    % returns a black image with white pixels of size 'sz' according to
    % coordinates in 'pts'.
    im = false(sz(1), sz(2));
    rows = offsets(2) + pts(:, 1);
    cols = offsets(1) + pts(:, 2);
    im(sub2ind(sz, rows, cols)) = true;
end

function ordered_points = order_points_along_line(points)
    % Orders a 3D point cloud representing a line
    %   points: Nx3 matrix of 3D points
    %   ordered_points: Nx3 matrix of sorted points along the principal axis
    
    % --- Perform PCA to find the principal axis ---
    % coeff: principal component directions (columns)
    % score: projections of points onto principal components
    [coeff, ~, ~] = pca(points);
    
    % Principal direction is the first component
    principal_axis = coeff(:,1);
    
    % --- Project points onto the principal axis ---
    projections = points * principal_axis;
    
    % --- Sort points based on their projection values ---
    [~, sorted_indices] = sort(projections);
    ordered_points = points(sorted_indices, :);
end

% function [base_pixels, tip_pixels, polyfit_base, base_fit_dim, polyfit_tip, tip_fit_dim] = fitToPoints(img, end_pixels, ord, numPtsFit, plotFlag)
function new_points = extend_centerlines_in_mask(mask, centerline_points, ord)
    % Extends centerline points to the ends of the mask by fitting a polynomial of order 'ord' onto points.
    
    new_points = centerline_points;
    num_pts_to_fit = min(size(centerline_points, 1), 50);
    % fit to both ends
    for i=1:2
        if i == 1
            pts_to_fit = centerline_points(1:num_pts_to_fit, :);
            end_point = centerline_points(1, :);
        else
            pts_to_fit = centerline_points(end-num_pts_to_fit+1:end, :);
            end_point = centerline_points(end, :);
        end
        
        % fit points to a line of 'ord'
        % could have ill-conditioned fit with all pixels not the same
        % e.g., too little points or straight line without all same points
        if min(pts_to_fit(:, 1)) ~= max(pts_to_fit(:, 1))
            polyfit_end = polyfit(pts_to_fit(:, 1), pts_to_fit(:, 2), ord);
            fit_dim = 0;
        elseif min(pts_to_fit(:, 2)) ~= max(pts_to_fit(:, 2))
            polyfit_end = polyfit(pts_to_fit(:, 2), pts_to_fit(:, 1), ord);
            fit_dim = 1;
        end
        
        % get extended points
        new_pixels = extend_centerline_in_mask(mask, pts_to_fit, end_point, polyfit_end, fit_dim);
        % remove some end points to get to center of tube
        num_pts_to_remove = 3;
        if size(new_pixels, 1) > 3
            new_pixels(end-(num_pts_to_remove-1):end, :) = [];
        end

        % add extended point to centerline
        if i == 1
            new_points = [flipud(new_pixels); new_points];
        else
            new_points = [new_points; new_pixels];
        end
    end
end


function new_pixels = extend_centerline_in_mask(mask, fitted_pixels, end_pixel, poly_coeffs, fit_dim)
    % Helper to extend skeleton to end of edge.
    % images 'skeleton' and 'mask' must be same size.
    
    % determine which direction to go towards
    img_size = size(mask);
    dirs = zeros(2, 1);
    if end_pixel(1) == 1 || ismember(end_pixel(1)-1, fitted_pixels(:, 1))
        dirs(1) = 1;
    elseif end_pixel(1) == img_size(2) || ismember(end_pixel(1)+1, fitted_pixels(:, 1))
        dirs(1) = -1;
    end
    if end_pixel(2) == 1 || ismember(end_pixel(2)-1, fitted_pixels(:, 2))
        dirs(2) = 1;
    elseif end_pixel(2) == img_size(1) || ismember(end_pixel(2)+1, fitted_pixels(:, 2))
        dirs(2) = -1;
    end
    
    if ~any(dirs)
        % should not happen
        fprintf("All dirs are 0!\n");
    end

    % using fitted polynomial 'poly', extend the fit to end of image
    % assumes that polynomial is a function (e.g., no loops)
    new_pixels = [];
    curr_pixel = end_pixel;
    % add pixels if they are part of tube (are white)
    while curr_pixel(1) >= 1 && curr_pixel(1) <= img_size(1) ...
            && curr_pixel(2) >= 1 && curr_pixel(2) <= img_size(2) ...
            && mask(curr_pixel(1), curr_pixel(2)) == 1
        % save new point
        new_pixels = [new_pixels; curr_pixel];
        
        % get candidate pixels
        if dirs(1) == 0
            candidate_pixels = [curr_pixel(1) - 1, curr_pixel(2) + dirs(2);
                curr_pixel(1), curr_pixel(2) + dirs(2);
                    curr_pixel(1) + 1, curr_pixel(2) + dirs(2)];
        elseif dirs(2) == 0
            candidate_pixels = [curr_pixel(1) + dirs(1), curr_pixel(2) - 1;
                curr_pixel(1) + dirs(1), curr_pixel(2);
                    curr_pixel(1) + dirs(1), curr_pixel(2) + 1];
        else
            candidate_pixels = [curr_pixel(1) + dirs(1), curr_pixel(2);
                curr_pixel(1), curr_pixel(2) + dirs(2);
                    curr_pixel(1) + dirs(1), curr_pixel(2) + dirs(2)];
        end

        % find closest pixel to the fit
        distances = Inf * ones(size(candidate_pixels, 1), 1);
        for i = 1:size(candidate_pixels, 1)
            if fit_dim == 0
                % use x
                fit_value = polyval(poly_coeffs, candidate_pixels(i, 1));
                distances(i) = abs(candidate_pixels(i, 2) - fit_value);
            else
                % use y
                fit_value = polyval(poly_coeffs, candidate_pixels(i, 2));
                distances(i) = abs(candidate_pixels(i, 1) - fit_value);
            end
        end

        % closest pixel is new current pixel
        [~, min_ind] = min(distances);
        curr_pixel = candidate_pixels(min_ind, :);
    end
end


function [chosen_point, chosen_ind] = choose_closer_point(point, candidates)
    % Choose the closest point in 'candidates' to 'point'
    %   point:      1xN vector (e.g., 1x3)
    %   candidates: MxN matrix of candidate points
    %
    %   chosen_point: 1xN closest point
    %   chosen_ind:   index of closest point (1-based)

    % Compute Euclidean distances to all candidates
    diffs = candidates - point;                    % MxN
    dists = sqrt(sum(diffs.^2, 2));                % Mx1

    % Find minimum
    [~, chosen_ind] = min(dists);
    chosen_point = candidates(chosen_ind, :);
end


function [world_points, img2_intersection_pts] = get_world_points(stereoParams, centerline_points, right_image)
    % Return world points (in space) from two skeleton stereoimages.
    % Finds epipolar correspondences, then batches triangulate for speed.
    img_size = size(right_image);
    num_centerline_pts = size(centerline_points, 1);
    left_pts = zeros(num_centerline_pts, 2);
    right_pts = zeros(num_centerline_pts, 2);
    found = false(num_centerline_pts, 1);
    x_all = 1:img_size(2);

    for sk = 1:num_centerline_pts
        pt = centerline_points(sk, :);
        l_pr = stereoParams.FundamentalMatrix * [pt(2); pt(1); 1];

        y_exact = -(l_pr(3) + l_pr(1) * x_all) / l_pr(2);
        y1 = floor(y_exact);
        y2 = ceil(y_exact);

        valid1 = y1 >= 1 & y1 <= img_size(1);
        hit1 = valid1;
        hit1(valid1) = right_image(sub2ind(img_size, y1(valid1), x_all(valid1)));

        if any(hit1)
            idx = find(hit1, 1);
            left_pts(sk, :) = [pt(2), pt(1)];
            right_pts(sk, :) = [x_all(idx), y1(idx)];
            found(sk) = true;
            continue;
        end

        valid2 = y2 >= 1 & y2 <= img_size(1);
        hit2 = valid2;
        hit2(valid2) = right_image(sub2ind(img_size, y2(valid2), x_all(valid2)));

        if any(hit2)
            idx = find(hit2, 1);
            left_pts(sk, :) = [pt(2), pt(1)];
            right_pts(sk, :) = [x_all(idx), y2(idx)];
            found(sk) = true;
        end
    end

    % Batch triangulate all matched points at once
    world_points = triangulate(left_pts(found, :), right_pts(found, :), stereoParams);
    img2_intersection_pts = [right_pts(found, 2), right_pts(found, 1)];
end



function aligned_points = align_points_to_origin(points, frame)
    % Align points to the origin using a frame
    %   points: Nx3 matrix
    %   frame:  4x4 homogeneous transform (rotation + translation)
    R = frame(1:3, 1:3);
    t = frame(1:3, 4);
    aligned_points = align_to_rotation( align_to_translation(points, t), R );
end


function aligned_points = align_to_rotation(points, R)
    %  Rotate points by R^T (inverse of R)
    %   points: Nx3
    %   R: 3x3 rotation matrix
    aligned_points = (R.' * points.').';
end


function aligned_points = align_to_translation(points, t)
    %  Translate points by subtracting t
    %   points: Nx3
    %   t: 3x1 or 1x3 vector
    aligned_points = points - t.';
end

function [splines_coefs, spline_breaks, control_pts] = world_pts2spline(world_pts, num_splines)
    % Returns spline characteristics given a sorted array of points.

    % if not a lot of points
    if size(world_pts, 1) < 20
        control_inds = [1, size(world_pts, 1)];
        control_pts = world_pts(control_inds, :);
        spline_fit = cscvn(control_pts');
        world_pts = get_spline_points(1, spline_fit.breaks, spline_fit.coefs);
    end

    % select points in array
    control_inds = floor(linspace(1, size(world_pts, 1), num_splines+1));
    control_pts = world_pts(control_inds, :);
    % use control points to generate splines
    spline_fit = cscvn(control_pts');

    % want to normalize break points to be from [0,1], [1,2] ... [num_splines-1, num_splines]
    spline_breaks = 0:1:num_splines;
    % get spline coefficients
    splines_coefs = spline_fit.coefs;  % each spline has 3 rows (x,y,z), each row has 4 coefs
    for i = 1:num_splines
        t1 = spline_fit.breaks(i);
        t2 = spline_fit.breaks(i+1);
        % rescale coefs from [t1, t2] to [i-1, i]
        for dim = 1:3
            % new coefs
            splines_coefs(3*(i-1)+dim, 1) = splines_coefs(3*(i-1)+dim, 1) * (t2 - t1)^3;
            splines_coefs(3*(i-1)+dim, 2) = splines_coefs(3*(i-1)+dim, 2) * (t2 - t1)^2;
            splines_coefs(3*(i-1)+dim, 3) = splines_coefs(3*(i-1)+dim, 3) * (t2 - t1);
            splines_coefs(3*(i-1)+dim, 4) = splines_coefs(3*(i-1)+dim, 4);
        end
    end

end

function full_spline_pts = get_spline_points(num_splines, spline_breaks, spline_coefs)
    % Returns spline characteristics given a sorted array of points.
    n = 100;
    full_spline_pts = zeros(n*num_splines, 3);
    for i = 1:num_splines
        t1 = spline_breaks(i);
        t2 = spline_breaks(i+1);
        t = linspace(t1, t2, n);
        fitpts = [spline_coefs(3*(i-1)+1, 1)*(t-t1).^3 + spline_coefs(3*(i-1)+1, 2)*(t-t1).^2 + spline_coefs(3*(i-1)+1, 3)*(t-t1) + spline_coefs(3*(i-1)+1, 4);
            spline_coefs(3*(i-1)+2, 1)*(t-t1).^3 + spline_coefs(3*(i-1)+2, 2)*(t-t1).^2 + spline_coefs(3*(i-1)+2, 3)*(t-t1) + spline_coefs(3*(i-1)+2, 4);
            spline_coefs(3*(i-1)+3, 1)*(t-t1).^3 + spline_coefs(3*(i-1)+3, 2)*(t-t1).^2 + spline_coefs(3*(i-1)+3, 3)*(t-t1) + spline_coefs(3*(i-1)+3, 4)];
        full_spline_pts((i-1)*n+1:i*n, :) = fitpts';
    end
end

function skel_full = crop_bwskel(mask)
    % Run bwskel on the bounding box of the mask, then place back in full image.
    [r, c] = find(mask);
    r1 = min(r); r2 = max(r);
    c1 = min(c); c2 = max(c);
    skel_crop = bwskel(mask(r1:r2, c1:c2));
    skel_full = false(size(mask));
    skel_full(r1:r2, c1:c2) = skel_crop;
end