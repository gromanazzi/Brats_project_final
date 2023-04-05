path_img = "..\preprocessedDataset\imagesTest";
path_labelsGT = "..\preprocessedDataset\labelsTest";
path_labelsPred = "..\output\labelsPredict";

%selezione rete
waitfor(msgbox('Select the net'));
[file, path]=uigetfile("*.mat", "Select the net","..\trained_nets");
load(fullfile(path, file));
cust_ext = split(file, '-');
cust_ext = cust_ext{end};

%selezione immagine e label
waitfor(msgbox('Select the volume'));
[file, path]=uigetfile("*.mat", "Select the volume", path_img);
img = load(fullfile(path, file));
[~, name, ~] = fileparts(file);
myfile = strcat(name, "-", cust_ext);

if ~exist(path_labelsPred,'dir')
    mkdir(fullfile(path_labelsPred));
end

if exist(fullfile(path_labelsGT, file), "file")
    label = load(fullfile(path_labelsGT, file));
    waitfor(msgbox('Ground Truth Label and volume successfully loaded'));
else 
    waitfor(msgbox('Ground Truth Label not found, please select it manually'));
    [file, path] = uigetfile("*.mat", "Select the label", path_labelsGT);
    label = load(fullfile(path, file));
end

global volTest;
global volTestLabels;
global predictedLabels;
volTest = img.cropVol;
volTestLabels = label.cropLabel;

if exist(fullfile(path_labelsPred, myfile), "file") && strcmp(questdlg('Predicted labels found. Do you want to use it?', 'Attention', 'Yes', 'No', 'Yes'), 'Yes')
    % Caricamento label pre-esistente
    predictedLabels = load(fullfile(path_labelsPred, myfile));
    predictedLabels = predictedLabels.predictedLabels;
else
    % Segmentazione
    msgbox('Segmentation started');
    bim = blockedImage(volTest);
    semanticsegBlock = @(bstruct)semanticseg(bstruct.Data,net);
    
    networkInputSize = net.Layers(1).InputSize;
    networkOutputSize = net.Layers(strcmpi({net.Layers.Name},"output")).OutputSize;
    
    blockSize = [networkOutputSize(1:3) networkInputSize(end)];
    borderSize = (networkInputSize(1:3) - blockSize(1:3))/2;
    
    batchSize = 1;

    results = apply(bim, ...
        semanticsegBlock, ...
        BlockSize=blockSize, ...
        BorderSize=borderSize,...
        PadPartialBlocks=true, ...
        BatchSize=batchSize);
    predictedLabels = results.Source;
    save(fullfile(path_labelsPred, myfile),"predictedLabels");
end

% Stampa risultati
global zID;
global zSliceGT;
global zSlicePred;
global slices_tot;

slices_tot = size(volTest,3);
zID = size(volTest,3)/2;
zSliceGT = labeloverlay(volTest(:,:,zID),volTestLabels(:,:,zID));
zSlicePred = labeloverlay(volTest(:,:,zID),predictedLabels(:,:,zID));
waitfor(msgbox(["Segmentation complete"; ...
    "Use up and down arrow on the keyboard to navigate trough the slices"]));fig = figure;
montage({zSliceGT,zSlicePred},Size=[1 2],BorderSize=5);

set(fig,'WindowKeyPressFcn',@KeyPressCb);
title("Labeled Ground Truth (Left) vs. Network Prediction (Right)")

% Modifica indice Z per scorrimento slice
function KeyPressCb(~,evnt)
        global zID;
        global zSliceGT;
        global zSlicePred;
        global volTest;
        global volTestLabels;
        global predictedLabels;
        global slices_tot;

        if strcmp(evnt.Key,'uparrow')==1 && zID<slices_tot
            zID = zID + 1;
            zSliceGT = labeloverlay(volTest(:,:,zID),volTestLabels(:,:,zID));
            zSlicePred = labeloverlay(volTest(:,:,zID),predictedLabels(:,:,zID));
            montage({zSliceGT,zSlicePred},Size=[1 2],BorderSize=5);
        elseif strcmp(evnt.Key, 'downarrow')==1 && zID>1
            zID = zID - 1;
            zSliceGT = labeloverlay(volTest(:,:,zID),volTestLabels(:,:,zID));
            zSlicePred = labeloverlay(volTest(:,:,zID),predictedLabels(:,:,zID));
            montage({zSliceGT,zSlicePred},Size=[1 2],BorderSize=5);
        end
end
