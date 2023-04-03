function blockMetrics = calculateBlockMetrics(bstruct,gtBlockLabels,net)

% Segmenta il blocco 
predBlockLabels = semanticseg(bstruct.Data,net);

% Rimozione bordo label 
blockStart = bstruct.BorderSize + 1;
blockEnd = blockStart + bstruct.BlockSize - 1;
gtBlockLabels = gtBlockLabels( ...
    blockStart(1):blockEnd(1), ...
    blockStart(2):blockEnd(2), ...
    blockStart(3):blockEnd(3));

% Valutazione risultati della segmentazione rispetto a quelli reali
confusionMat = segmentationConfusionMatrix(predBlockLabels,gtBlockLabels);

% Creazione struttura blockMetrics con matrice di confusione, 
% numero di immaginiblockMetrics e  informazioni sul blocco. 
blockMetrics.ConfusionMatrix = confusionMat;
blockMetrics.ImageNumber = bstruct.ImageNumber;
blockInfo.Start = bstruct.Start;
blockInfo.End = bstruct.End;
blockMetrics.BlockInfo = blockInfo;

end