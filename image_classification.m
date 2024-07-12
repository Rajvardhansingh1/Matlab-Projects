images = imageDatastore("Hnd", 'IncludeSubfolders', true, 'LabelSource','foldernames');
numFiles = numel(images.Files);

%for k = 1:numFiles
%    img = readimage(images,k);
%    [rows,cols,channels] = size(img);
%end
%disp(['Number of Images : ', num2str(numel(images.files))]);
%disp(['Number of Images L ',num2str(numel(unique(images.labels)))]);

trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;

[imageTrain, imageRem] = splitEachLabel(images,trainRatio,'Randomized');
[imageVal, imageTest] = splitEachLabel(imageRem, valRatio/(valRatio + testRatio), 'Randomized');
totalImages = numel(imageTrain.Files) + numel(imageVal.Files) + numel(imageTest.Files);
fprintf("%d\n",totalImages);
fprintf("%d\n", numel(imageVal.Files));
fprintf("%d\n", numel(imageTest.Files));
fprintf("%d\n", numel(imageTrain.Files));

uniqueLabels = unique(images.Labels);

% % Create a figure to display the images
% figure;
% 
% % Define the number of images to display per class (if you want to limit the number)
% numImagesToDisplay = 5;
% 
% % Iterate through each label
% for i = 1:numel(uniqueLabels)
%     label = uniqueLabels(i);
% 
%     % Find indices of images with the current label
%     labelIndices = find(images.Labels == label);
% 
%     % Display images with the current label
%     for j = 1:min(numImagesToDisplay, numel(labelIndices))
%         subplot(numel(uniqueLabels), numImagesToDisplay, (i-1)*numImagesToDisplay + j);
%         img = readimage(images, labelIndices(j));
%         imshow(img);
%         title(char(label));
%     end
% end

Layers = [
    imageInputLayer([28 28 1], 'Name','input')
    convolution2dLayer(3, 64, 'padding','same', 'Name', 'convo1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name','relu1')
    dropoutLayer(0.25, 'Name', 'dropout1')
    maxPooling2dLayer(2,'stride',2,'Name','Maxp1')
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2') 
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.50, 'Name', 'dropout2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4') 
    batchNormalizationLayer('Name', 'batchnorm4')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.50, 'Name', 'dropout4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4')

    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.25, 'Name', 'dropout5')
    fullyConnectedLayer(numel(unique(images.Labels)), 'Name', 'fc') 
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
    ];
disp(Layers);

augmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ...
    'RandXTranslation',[-3,3], ...
    'RandYTranslation',[-3,3]);
augimdsTrain = augmentedImageDatastore([28 28 1], imageTrain, 'DataAugmentation', augmenter);

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10, ...
    'MaxEpochs', 35, ...
    'MiniBatchSize', 32, ...
    'ValidationData', imageVal, ...
    'ValidationFrequency', 15, ...
    'ExecutionEnvironment', 'gpu', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(imageTrain,Layers,options);
YPred = classify(net, imageTest);
YTest = imageTest.Labels;

accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Test accuracy: ', num2str(accuracy)]);

figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');

save("Image_Classification.mat");