path_img = "..\preprocessedDataset\imagesTest";
path_labelsGT = "..\preprocessedDataset\labelsTest";
path_metrics = "..\output\metrics";

if ~exist(path_metrics,'dir')
    mkdir(fullfile(path_metrics));
end

answer = questdlg('Effettuare la valutazione su', ...
	'Validazione', ...
    'Singolo File','Cartella Test', ...
    'Cartella Test');

switch answer
    case 'Singolo File'
        [file, path] = uigetfile("*.mat", "Seleziona l'immagine (in formato mat)","..\preprocessedDataset\imagesTest");
        volLocTest = fullfile(path, file);
        if exist(fullfile(path_labelsGT, file), "file")
            lblLocTest = fullfile(path_labelsGT, file);
            waitfor(msgbox('Label and volume successfully loaded'));
        else 
            waitfor(msgbox('Label not found, please select manually'));
            [file, path] = uigetfile("*.mat", "Select the label", path_labelsGT);
            lblLocTest = fullfile(path, file);
        end

    case 'Cartella Test'
        volLocTest = fullfile(path_img);
        lblLocTest = fullfile(path_labelsGT);
end

[file, path]=uigetfile("*.mat", "Select the net","..\trained_nets");
load(fullfile(path, file));
cust_ext = split(file, '-');
cust_ext = cust_ext{end};


classNames = ["background","tumor"];
pixelLabelID = [0 1];

%Creazione datastore per salvare volumi e label
voldsTest = imageDatastore(volLocTest,FileExtensions=".mat", ...
    ReadFcn=@matRead);
pxdsTest = pixelLabelDatastore(lblLocTest,classNames,pixelLabelID, ...
    FileExtensions=".mat",ReadFcn=@matRead);

imageIdx = 1;
datasetConfMat = table;

while hasdata(voldsTest)
    [filepath,name,ext] = fileparts(voldsTest.Files(imageIdx));

    % Lettura dei dati dei volumi e delle labels
    vol = read(voldsTest);
    volLabels = read(pxdsTest);
    if exist (fullfile(path_metrics, strcat(name, "-", cust_ext)), "file")
        blockConfMatOneImage = load(fullfile(path_metrics, strcat(name, "-", cust_ext)));
        blockConfMatOneImage = blockConfMatOneImage.blockConfMatOneImage;
    else
        % Creazione blockedImage per i dati del volume e label 
        testVolume = blockedImage(vol);
        testLabels = blockedImage(volLabels{1});
    
        networkInputSize = net.Layers(1).InputSize;
        networkOutputSize = net.Layers(strcmpi({net.Layers.Name},"output")).OutputSize;
        blockSize = [networkOutputSize(1:3) networkInputSize(end)];
        borderSize = (networkInputSize(1:3) - blockSize(1:3))/2;
        
        % Calcolo metriche
        blockConfMatOneImage = apply(testVolume, ...
            @(block,labeledBlock) ...
                calculateBlockMetrics(block,labeledBlock,net), ...
            ExtraImages=testLabels, ...
            PadPartialBlocks=true, ...
            BlockSize=blockSize, ...
            BorderSize=borderSize);

        save(fullfile(path_metrics, strcat(name, "-", cust_ext)), "blockConfMatOneImage")
    end

    % Legge i risultati ottenuti su un'immagine e aggiorna il numero dell'immagine
    blockConfMatOneImageDS = blockedImageDatastore(blockConfMatOneImage);
    blockConfMat = readall(blockConfMatOneImageDS);
    blockConfMat = struct2table([blockConfMat{:}]);
    blockConfMat.ImageNumber = imageIdx.*ones(height(blockConfMat),1);
    datasetConfMat = [datasetConfMat;blockConfMat]; %#ok<AGROW> 
    
    fprintf("Iteration %d of %d complete\n", imageIdx, size(voldsTest.Files, 1));
    imageIdx = imageIdx + 1;
end

[metrics,blockMetrics] = evaluateSemanticSegmentation( ...
    datasetConfMat,classNames,Metrics="all");