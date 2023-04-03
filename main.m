% Main
% Viene chiesto all'utente di selezionare il tipo di operazione da
% effettuare e vengono richiamate le relative classi/funzioni
src_folder = 'src';
out_dir = 'output';
if ~exist(out_dir,'dir')
    mkdir(out_dir);
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