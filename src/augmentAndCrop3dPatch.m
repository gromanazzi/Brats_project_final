
%Gestisce in modo casuale i dati di addestramento per rendere l'addestramento più robusto.
% La funzione non modifica i dati di validazione
function patchOut = augmentAndCrop3dPatch(patchIn,outPatchSize,flag)

isValidationData = strcmp(flag,'Validation');

% Crea due celle vuote di dimensione size(patchIn,1),1), cioè la prima
% dimensione dell'immagine di ingresso patchIn e 1
inpVol = cell(size(patchIn,1),1);
inpResponse = cell(size(patchIn,1),1);

%% 5 augmentations: nil,rot90,fliplr,flipud,rot90(fliplr)

% fliprot è una funzione anonima che esegue la rotazione di 90 gradi
% dell'immagine specchiata con fliplr
fliprot = @(x) rot90(fliplr(x));
% in augtype abbiamo 4 funzioni: rotazione 90 antioraria, riflessione 
% left-right, riflessione up-down, fliprot (definita prima)
augType = {@rot90,@fliplr,@flipud,fliprot};
% Ciclo for lungo tutti i valori contenuti nella patch di ingresso
for id=1:size(patchIn,1) 
    % Selezione di un int random tra 1 e 8
    rndIdx = randi(8,1);
    % Estrazione dell'immagine e della label dal dataset iniziale
    tmpImg =  patchIn.InputImage{id};
    tmpResp = patchIn.ResponsePixelLabelImage{id};
    % Se abbiamo un int>4 o sono dati di validazione non facciamo niente,
    % altrimenti facciamo una modifica
    if rndIdx > 4 || isValidationData
        out =  tmpImg;
        respOut = tmpResp;
    else
        out =  augType{rndIdx}(tmpImg);
        respOut = augType{rndIdx}(tmpResp);
    end
    % Crop the response to to the network's output.
    % Legge i valori delle 3 dimensioni dell'immagine uscita dal ciclo for 
    % e del layer di uscita, 
    inSize = size(respOut, [1 2 3]);
    outSize = outPatchSize(1:3);
    % Calcola le coordinate dell'immagine da ritagliare affinché l'immagine
    % inSize abbia alla fine le dimensioni outSize
    win = centerCropWindow3d(inSize, outSize);
    % Usa la funzione imcrop3d per effettuare un ritaglio (3D)
    % dell'immagine respOut usando le dimensioni contenute in win
    respFinal = imcrop3(respOut, win);
    % Inserisce le immagini e le label nelle celle inpVol e inpResponse,
    % per poi metterle nella tabella patchOut, restituita come output dalla
    % funzione
    inpVol{id,1} = out;
    inpResponse{id,1} = respFinal;
end
patchOut = table(inpVol,inpResponse);