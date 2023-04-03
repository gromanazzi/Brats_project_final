%prova di utilizzo di una rete preaddestrata

%cartella di lavoro
dataDir = fullfile("..\");
if ~exist(fullfile(dataDir,"trained_nets"), 'dir')
    mkdir(fullfile(dataDir,"trained_nets"));
end 

%download rete neurale
url = "https://www.mathworks.com/supportfiles/vision/data/brainTumor3DUNetValid.mat";
[~,name,filetype] = fileparts(url);
netFileFullPath = fullfile(dataDir,"trained_nets",strcat("trained3DUNet-Example",filetype));

if not(exist(netFileFullPath,"file"))
    disp("Downloading pretrained network.");
    disp("This can take several minutes to download...");
    websave(netFileFullPath,url);

    if filetype == ".zip"
        unzip(netFileFullPath,dataDir)
    end
    disp("Done.");
else
    disp("U-Net already present")
end
%fine download rete

%carica i dati in memoria
load(netFileFullPath);

%download esempi
sampledata_url = "https://www.mathworks.com/supportfiles/vision/data/sampleBraTSTestSetValid.tar.gz";
imageDataLocation = fullfile(dataDir,'sampleBraTSTestSetValid');
if ~exist(imageDataLocation, 'dir')
    fprintf('Downloading sample BraTS test data set.\n');
    %fprintf('This will take several minutes to download and unzip...\n');
    untar(sampledata_url,dataDir);
    fprintf('File Download complete.\n\n');
else
    fprintf("Examples already present\n");
end
%fine download esempi

testDir = fullfile(dataDir,'sampleBraTSTestSetValid');
data = load(fullfile(testDir, "imagesTest/", "BraTS446.mat"));
labels = load(fullfile(testDir, "labelsTest/", "BraTS446.mat"));

volTest = data.cropVol;
volTestLabels = labels.cropLabel;

% segmentazione
bim = blockedImage(volTest);
semanticsegBlock = @(bstruct)semanticseg(bstruct.Data,net);

networkInputSize = net.Layers(1).InputSize;
networkOutputSize = net.Layers(strcmpi({net.Layers.Name},"output")).OutputSize;

blockSize = [networkOutputSize(1:3) networkInputSize(end)];
borderSize = (networkInputSize(1:3) - blockSize(1:3))/2;

results = apply(bim, ...
    semanticsegBlock, ...
    BlockSize=blockSize, ...
    BorderSize=borderSize,...
    PadPartialBlocks=true, ...
    BatchSize=batchSize);
predictedLabels = results.Source;

% Stampa
zID = size(volTest,3)/2;
zSliceGT = labeloverlay(volTest(:,:,zID),volTestLabels(:,:,zID));
zSlicePred = labeloverlay(volTest(:,:,zID),predictedLabels(:,:,zID));

fig = figure;
hShow = montage({zSliceGT,zSlicePred},Size=[1 2],BorderSize=5);
title("Labeled Ground Truth (Left) vs. Network Prediction (Right)")
