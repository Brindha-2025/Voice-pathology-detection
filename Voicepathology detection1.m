%add folders to the path
addpath(genpath('men'))
addpath(genpath('women'))

 %load speech signal and create scalogram images
[FileName,FilePath]=uigetfile('*.wav','Select One or More Files', ...
   'MultiSelect', 'on');
for i=1:size(FileName,2)
    fn=fullfile(FilePath,FileName{1,i});
    [aud,fs]=audioread(fn);
    aud1{i}=aud';
    fs(:,i)=fs;
    img1{i}=cwt(aud1{i},fs(:,i));
end
 
%load EGG signal and create scalogram images
 [FileName1,FilePath1]=uigetfile('*.wav','Select One or More Files', ...
    'MultiSelect', 'on');
 for i1=1:size(FileName1,2)
          fn1=fullfile(FilePath1,FileName1{1,i1});
 [aud21,fs1]=audioread(fn1);
     aud2{i1}=aud21';
     fs1(:,i1)=fs1;
     img2{i1}=cwt(aud2{i1},fs1(:,i1));
 end

%********************************************
input=[img1 img2];
label1=ones(1,size(img1,2)); label2=zeros(1,size(img1,2));
labels=[label1 label2];

imgSize = [128, 128];
numImages = numel(input);
processedImages = zeros(imgSize(1), imgSize(2), 1, numImages); %

for i = 1:numImages
    img = input{i};
    imgResized = imresize(img, imgSize); 

    magnitude = abs(imgResized); 
    phase = angle(imgResized);   
    
    
    processedImages(:, :, 1, i) = magnitude;
    processedImages(:, :, 2, i) = phase;
end

X = processedImages; 
Y = categorical(labels); 


numImages = size(processedImages, 4);


idx = randperm(numImages);
numTrain = round(0.8 * numImages);
numVal = round(0.1 * numImages);
numTest = numImages - numTrain - numVal;


trainIdx = idx(1:numTrain);
valIdx = idx(numTrain+1:numTrain+numVal);
testIdx = idx(numTrain+numVal+1:end);


XTrain = processedImages(:,:,:,trainIdx);
YTrain = Y(trainIdx);

XVal = processedImages(:,:,:,valIdx);
YVal = Y(valIdx);

XTest = processedImages(:,:,:,testIdx);
YTest = Y(testIdx);


layers = [
    imageInputLayer([imgSize 2], 'Name', 'input') 
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(2, 'Name', 'fc') 
    ];


options = trainingOptions('sgdm', ...
    'MaxEpochs',5, ...
    'Verbose',false, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress');


net = trainNetwork(XTrain, YTrain,layers, options);


layerName = 'fc';


featuresTrain = activations(net, XTrain, layerName, 'OutputAs', 'rows');
labelsTrain = imdsTrain.Labels;


featuresTest = activations(net,  XTest, layerName, 'OutputAs', 'rows');
labelsTest = YTest;


svmModel = fitcecoc(featuresTrain, labelsTrain);


YPred = predict(svmModel, featuresTest);


accuracy = sum(predictedLabels == labelsTest) / numel(labelsTest);
fprintf('Test Accuracy using DNN-SVM: %.2f%%\n', accuracy * 100);


YPred1= str2double(string(YPred));

if (YPred1 == 1)
    re='Heathy';
elseif (YPred1==0)
    re='pathological';
else
    re='ivalid input';
end

 