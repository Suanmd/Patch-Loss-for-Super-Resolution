function evaluate_results_dirs(input_dir,GT_dir,shave_width,verbose)
%% Parameters
% Directory with your results
%%% Make sure the file names are as exactly %%%
%%% as the original ground truth images %%%
% input_dir = fullfile(pwd,'your_results');

% Directory with ground truth images
% GT_dir = fullfile(pwd,'self_validation_HR');

% Number of pixels to shave off image borders when calcualting scores
% shave_width = 2;

% Set verbose option
% verbose = true;
addpath utils

input_dir_SR = dir([input_dir, '\visualization']);
for kk = 1:size(input_dir_SR, 1)
    if length(input_dir_SR(kk).name) >= 10
        %% Make SRdir and HRdir 
        SR_dir_name = input_dir_SR(kk).name;
        gt_name = strsplit(input_dir_SR(kk).name, '_');
        HR_dir_name = char(gt_name(1));
        SR_dir = fullfile(input_dir, 'visualization', SR_dir_name);
        HR_dir = fullfile(GT_dir, HR_dir_name, 'GTmod12');
        addpath(SR_dir)
        addpath(HR_dir)
        %% Calculate scores and save
        scores = calc_scores(SR_dir, HR_dir, str2num(shave_width), verbose);
        %% Printing results
        % perceptual_score = (mean([scores.NIQE]) + (10 - mean([scores.Ma]))) / 2;
        % fprintf(['\n\nYour perceptual score is: ',num2str(perceptual_score)]);
        % fprintf(['\nYour RMSE is: ',num2str(sqrt(mean([scores.MSE]))),'\n']);
        %% Rename
        for ii = 1:size(scores, 2)
            oldname = scores(ii).name;
            newname1 = oldname(1:end-4);
            newname3 = oldname(end-3:end);
            newname2 = ['_', num2str(scores(ii).NIQE), ...
                        '_', num2str(scores(ii).Ma), ...
                        '_', num2str(scores(ii).PI)];
            newname = [newname1, newname2, newname3];
            oldname2 = fullfile(SR_dir, scores(ii).name);
            eval(['!rename',' "',oldname2,'" ',newname])
        end
        foldername2 = ['_', num2str(mean([scores.NIQE])), ...
                       '_', num2str(mean([scores.Ma])), ...
                       '_', num2str(mean([scores.PI]))];
        new_SR_dir_name = [SR_dir_name, foldername2];
        eval(['!rename',' "',fullfile(input_dir, 'visualization', SR_dir_name), ...
              '" ',new_SR_dir_name])
    end
end