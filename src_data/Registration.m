close all
clear all
addpath('./stlTools/');

DatasetPath = './Dataset'; % Change to the dataset folder
RegistrationPath = './Registration Methods/Test'; % Change to the folder where your want to save registration output files

Patients = 4;
TREall = [];
ICall = [];
for PatientNo = 1:Patients
    LapPath = [DatasetPath, '/Patient', num2str(PatientNo), '/Lap Images/'];
    modelPath = [DatasetPath, '/Patient', num2str(PatientNo), '/Preoperative Model/'];

    regPath = [RegistrationPath, '/Patient', num2str(PatientNo), '/'];
    if ~exist(regPath, 'dir')
        mkdir(regPath)
    end
    
    LapImgfiles = dir([LapPath,'*.png']);

    N = length(LapImgfiles); % Number of images

    [LiverPoints, LiverConnectivity, n, name] = stlRead([modelPath, 'Liver.stl']);    
    [TumourPoints, TumourConnectivity, n, name] = stlRead([modelPath, 'Tumour.stl']);    
    
    for i=1:N
        number = LapImgfiles(i).name;
        number = number(1:end-4);
        LapImg = imread([LapPath, LapImgfiles(i).name]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Add your code here to register preoperative liver model in
        %%% (LiverPoints, LiverConnectivity) on the laparoscopic image in
        %%% LapImg and report the registered tumour model in
        %%% (RegTumourPoints, RegTumourConnectivity).
        RegTumourConnectivity = TumourConnectivity; %Remove this line
        RegTumourPoints = TumourPoints; %Remove this line
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        OutputSTLfile = [regPath,number, '.stl'];
        stlWrite(OutputSTLfile, RegTumourConnectivity, RegTumourPoints);
    end    
end