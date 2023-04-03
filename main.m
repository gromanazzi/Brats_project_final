% Main
% Viene chiesto all'utente di selezionare il tipo di operazione da
% effettuare e vengono richiamate le relative classi/funzioni
src_folder = 'src';
sourceDataLoc = "..\Task01_BrainTumour";
out_dir = 'output';
if ~exist(out_dir,'dir')
    mkdir(out_dir);
end 


preprocessDataLoc = "..\preprocessedDataset";
if ~exist(preprocessDataLoc,'dir')
    mkdir(preprocessDataLoc);
end

net_dir = "..\trained_nets";
if ~exist(net_dir,'dir')
    mkdir(net_dir);
end

path = dir(preprocessDataLoc);
numSubfolders = sum([path(:).isdir]) - 2;
origTrLoc = fullfile(sourceDataLoc, "imagesTr");
origTsLoc = fullfile(sourceDataLoc, "imagesTs");
volLoc = fullfile(preprocessDataLoc,"imagesTr");
lblLoc = fullfile(preprocessDataLoc,"labelsTr");
volLocVal = fullfile(preprocessDataLoc,"imagesVal");
lblLocVal = fullfile(preprocessDataLoc,"labelsVal");
volLocTest = fullfile(preprocessDataLoc,"imagesTest");
lblLocTest = fullfile(preprocessDataLoc,"labelsTest");

% Controllo sul numero di file totali, per controllare se effettivamente il
% preprocess è andato a buon fine
numNii = size(dir(fullfile(origTrLoc, "BRATS*.gz")),1);
numMat = size(dir(fullfile(volLoc, "*.mat")),1) + ...
    size(dir(fullfile(volLocVal, "*.mat")),1) + ...
    size(dir(fullfile(volLocTest, "*.mat")),1);
if numSubfolders ~= 6 || numNii > numMat
    resp = questdlg('It seems that preprocess is not complete. complete it now?', ...
        'Attention', 'Yes', 'No', 'Yes');
    if strcmpi(resp, 'yes')
        addpath(genpath(src_folder))
        preprocessBraTSDataset(preprocessDataLoc,sourceDataLoc);
    else
        error("Preprocess is mandatory to perform further operations");
    end
end

[idx, tf] = listdlg('PromptString', "Select the operation", ...
    'SelectionMode','single', ...
    'ListString', {'Train a new net','Segmentation', 'Performance Evaluation', 'Example'});
if tf
   switch idx
       case 1
           run(fullfile(src_folder, 'train.m'))
       case 2
           run(fullfile(src_folder, 'segmentation.m'))
       case 3
           run(fullfile(src_folder, 'evaluate.m'))
       case 4
           run(fullfile(src_folder, 'example.m'))
   end
end