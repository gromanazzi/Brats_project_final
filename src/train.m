sourceDataLoc = "..\Task01_BrainTumour";
preprocessDataLoc = "..\preprocessedDataset";

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
    preprocessBraTSDataset(preprocessDataLoc,sourceDataLoc);
end

%Creazione ImageDataStore per salvare le immagini 3D
volds = imageDatastore(volLoc,FileExtensions=".mat",ReadFcn=@matRead);

%Creazione pixelLabelDatastore per salvare le label
classNames = ["background","tumor"];
pixelLabelID = [0 1];
pxds = pixelLabelDatastore(lblLoc,classNames,pixelLabelID, ...
    FileExtensions=".mat",ReadFcn=@matRead);

%Creazione randomPatchExtractionDatastore che estrae patch casuali dalle
%immagini e label di training
patchSize = [132 132 132];
patchPerImage = 8;
% Minibatch di default impostato a 8, se usiamo una GPU per il training 
% andiamo Out Of Memory, quindi lo mettiamo a 2
miniBatchSize = 2; 
% RandomPatch estrae un numero definito dall'utente di immagini dal
% datastore di partenza (volds) e da quello di arrivo(pxds) al fine di
% avere pezzi randomici di immagini (patch) da dare alla 3dunet per
% addestrarsi. crea un datastore con tante righe (per immagine) quante
% sono indicate nel minibatchsize e 
patchds = randomPatchExtractionDatastore(volds,pxds,patchSize, ...
    PatchesPerImage=patchPerImage);
patchds.MiniBatchSize = miniBatchSize;

% Creazione randomPatchExtractionDatastore che estrae patch casuali dalle
% immagini e label di validazione

voldsVal = imageDatastore(volLocVal,FileExtensions=".mat", ...
    ReadFcn=@matRead);


pxdsVal = pixelLabelDatastore(lblLocVal,classNames,pixelLabelID, ...
    FileExtensions=".mat",ReadFcn=@matRead);

dsVal = randomPatchExtractionDatastore(voldsVal,pxdsVal,patchSize, ...
    PatchesPerImage=patchPerImage);
dsVal.MiniBatchSize = miniBatchSize;

%Creazione di una rete U-Net 3D
numChannels = 4;
inputPatchSize = [patchSize numChannels];
numClasses = 2;
[lgraph,outPatchSize] = unet3dLayers(inputPatchSize, ...
    numClasses,ConvolutionPadding="valid");

% Migliora i dati di training e validation tramite la funzione
% augmentAndCrop3d
dsTrain = transform(patchds, ...
    @(patchIn)augmentAndCrop3dPatch(patchIn,outPatchSize,"Training"));
dsVal = transform(dsVal, ...
    @(patchIn)augmentAndCrop3dPatch(patchIn,outPatchSize,"Validation"));

% Impostazione di un dicePixel layer per la classificazione dei pixel e la
% successiva previsione di appartenenza a una delle classi impostate
% precedentemente, sostituendo il segmentation layer usato di default
outputLayer = dicePixelClassificationLayer(Name="Output");
lgraph = replaceLayer(lgraph,"Segmentation-Layer",outputLayer);

% Creazione inputLayer senza normalizzazione in quanto i dati sono stati 
% normalizzati precedentemente (imageinputlayer di norma effettua la 
% normalizzazione che abbiamo fatto durante il preprocessing)
inputLayer = image3dInputLayer(inputPatchSize, ...
    Normalization="none",Name="ImageInputLayer");
lgraph = replaceLayer(lgraph,"ImageInputLayer",inputLayer);

% Impostazione opzioni per la rete U-Net
options = trainingOptions("adam", ...
    MaxEpochs=2, ...
    InitialLearnRate=5e-2, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=5, ...
    LearnRateDropFactor=0.95, ...
    ValidationData=dsVal, ...
    ValidationFrequency=50, ...
    Plots="training-progress", ...
    Verbose=true, ...
    MiniBatchSize=miniBatchSize, ...
    ExecutionEnvironment="auto");

% Aallenamento e salvataggio della rete
[net,info] = trainNetwork(dsTrain,lgraph,options);
modelDateTime = string(datetime("now",Format="yyyy-MM-dd"));
save(fullfile(net_dir,"trained3DUNet-"+modelDateTime+".mat","net"));