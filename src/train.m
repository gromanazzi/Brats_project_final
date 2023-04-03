sourceDataLoc = "..\Task01_BrainTumour";
preprocessDataLoc = "..\preprocessedDataset";

volLoc = fullfile(preprocessDataLoc,"imagesTr");
lblLoc = fullfile(preprocessDataLoc,"labelsTr");
volLocVal = fullfile(preprocessDataLoc,"imagesVal");
lblLocVal = fullfile(preprocessDataLoc,"labelsVal");

resp = questdlg('Proceed with net training?', 'Attention', 'Yes', 'No', 'No');
if strcmpi(resp, 'yes')
    
    classNames = ["background","tumor"];
    pixelLabelID = [0 1];
         
    
    % Creazione randomPatchExtractionDatastore che estrae patch casuali 
    % dalle immagini e label di training
    % RandomPatch estrae un numero definito dall'utente di immagini dal
    % datastore di partenza (volds) e da quello di arrivo(pxds) da dare 
    % alla 3dunet per addestrarsi.
    patchSize = [132 132 132];
    patchPerImage = 8;
    miniBatchSize = 2; 

    % Creazione ImageDataStore per salvare le immagini 3D e
    % pixelLabelDatastore per le label
    volds = imageDatastore(volLoc,FileExtensions=".mat",ReadFcn=@matRead);
    pxds = pixelLabelDatastore(lblLoc,classNames,pixelLabelID, ...
        FileExtensions=".mat",ReadFcn=@matRead); 

    patchds = randomPatchExtractionDatastore(volds,pxds,patchSize, ...
        PatchesPerImage=patchPerImage);
    patchds.MiniBatchSize = miniBatchSize;
    
    % Creazione randomPatchExtractionDatastore che estrae patch casuali 
    % dalle immagini e label di validazione    
    voldsVal = imageDatastore(volLocVal,FileExtensions=".mat", ...
        ReadFcn=@matRead);
    pxdsVal = pixelLabelDatastore(lblLocVal,classNames,pixelLabelID, ...
        FileExtensions=".mat",ReadFcn=@matRead);
    
    dsVal = randomPatchExtractionDatastore(voldsVal,pxdsVal,patchSize, ...
        PatchesPerImage=patchPerImage);
    dsVal.MiniBatchSize = miniBatchSize;
    
    %  Creazione di una rete U-Net 3D
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
    
    % Impostazione di un dicePixel layer per la classificazione dei pixel 
    outputLayer = dicePixelClassificationLayer(Name="Output");
    lgraph = replaceLayer(lgraph,"Segmentation-Layer",outputLayer);
    
    % Creazione inputLayer senza normalizzazione 
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
    
    % Allenamento e salvataggio della rete
    [net,info] = trainNetwork(dsTrain,lgraph,options);
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd"));
    save(fullfile(net_dir,"trained3DUNet-"+modelDateTime+".mat","net"));
end