close all
clear all
addpath('./stlTools/');

DatasetPath = './Dataset'; % Change to the dataset folder
RegistrationPath = './Registration Methods/Adagolodjo'; % Change to the folder where your registration output files are located

Patients = 4;
TREall = [];
ICall = [];
for PatientNo = 1:Patients
    pathUSprj = [DatasetPath, '/Patient', num2str(PatientNo), '/LUS Calibration and Pose/'];
    regPath = [RegistrationPath, '/Patient', num2str(PatientNo), '/'];
    USSegPath = [DatasetPath, '/Patient', num2str(PatientNo), '/LUS Segmentation/json/'];

    USprjMatfiles = dir([pathUSprj,'*.mat']);

    K = 10; % Number of iterations
    M = 50; % Number of samples over each profile
    N = length(USprjMatfiles); % Number of images

    %%%%% Read Patint data from dataset 
    TumourPoints = cell(1,N);
    TumourConnectivity = cell(1,N);
    P = zeros(3,M,N);
    for i=1:N
        number = USprjMatfiles(i).name;
        number = number(1:end-4);

        [Points, Connectivity, n, name] = stlRead([regPath, number, '.stl']);
        % Points(:,2:3) = -Points(:,2:3);  % Rotate 180 deg around X axis to change OpenGL CS to Matlab CS
        TumourPoints{i} = Points;
        TumourConnectivity{i} = Connectivity;
        jsonfileName = [USSegPath, number, '.json']; % filename in JSON extension
        fid = fopen(jsonfileName); % Opening the file
        raw = fread(fid,inf); % Reading the contents
        str = char(raw'); % Transformation
        fclose(fid); % Closing the file
        data = jsondecode(str); % Using the jsondecode function to parse JSON from string
        USprofile=(data.objects.points.exterior)';
        load([pathUSprj,number,'.mat']);
        USprofile = [SLus * USprofile(1,:); zeros(size(USprofile(1,:))); SAus * USprofile(2,:)];
        USprofile_probe = (Rus * USprofile) + Tus;
        USprofile_cam = (Rpr * USprofile_probe + Tpr);
        P(:,:,i) = SampleProfile(USprofile_cam,M); % Getting M samples from profile P
    end    
    
    TRE = TREcalc(P, TumourPoints, K);
    IC = ICcalc(P, TumourPoints, TumourConnectivity, 10);
    disp(['Patient ', num2str(PatientNo), ':']);    
    disp(['TRE = ', num2str(TRE)]);
    if IC
        disp('IC = Pass');
    else
        disp('IC = Failed');
    end
    disp(' ');
    TREall = [TREall, TRE];
    ICall = [ICall, IC];
end

disp(['Overall:']);
disp(['TRE = ', num2str(mean(TREall))]);
disp(['IC = ', num2str(sum(ICall)), ' of ', num2str(Patients)]);




% Measuring Inclusion Criterion
function IC = ICcalc(P, TumourPoints, TumourConnectivity, OncologicMargin)
N = length(TumourPoints);
IC = 1;
for i=1:N
    T = TumourPoints{i};
    Tc = mean(T);
    TumorPointsAug = zeros(size(T));
    for j=1:length(T)
        Point = T(j,:);
        TumorPointsAug(j,:) = Point + OncologicMargin * (Point-Tc)/vec_norm(Point-Tc);
    end    
    IC = IC & all(intriangulation(TumorPointsAug,TumourConnectivity{i},P(:,:,i)'));    
end
end

% Measuring TRE Criterion
function TRE = TREcalc(P, TumourPoints, K)
N = length(TumourPoints);
%%%%% Initialization
R0=eye(3);
t0=[];
for i=1:N
    CoG_P = mean(P(:,:,i),2);
    CoG_T = mean(TumourPoints{i}',2);
    t0 = [t0,CoG_T-CoG_P];
end
t0 = mean(t0,2);

% ICP  
R = R0;
t = t0;
for k=1:K
    T = ClosetPoint(TumourPoints, P, R, t);
    [regParams,Pfit,ErrorStats] = absor(P(:,:),T(:,:));
    R = regParams.R;
    t = regParams.t;
end

P = P(:,:);
T = T(:,:);
TRE = mean(vec_norm(P-T));
end

function PSampled = SampleProfile(P,M)
Perimeter = sum(vec_norm(P(:,2:end) - P(:,1:end-1)));
delta_P = Perimeter / M;
PSampled(:,1) = P(:,1);
Next = 2;
CurrentPoint = P(:,1);
NextPoint = P(:,2);
Distance2NextPoint = norm(NextPoint-CurrentPoint);
for i = 2:M
    delta_Pi = delta_P;
    while delta_Pi>Distance2NextPoint
        Next = Next + 1;
        delta_Pi = delta_Pi - Distance2NextPoint;
        CurrentPoint = P(:,Next-1);
        NextPoint = P(:,Next);
        Distance2NextPoint = norm(NextPoint-CurrentPoint);
    end
    CurrentPoint = CurrentPoint + delta_Pi * (NextPoint-CurrentPoint)./norm(NextPoint-CurrentPoint);
    Distance2NextPoint = norm(NextPoint-CurrentPoint);
    PSampled(:,i) = CurrentPoint;
end
end

function TCP = ClosetPoint(TumourPoints, P, R, t)
    M = size(P,2);
    N = size(P,3);
    TCP = zeros(3,M,N);
    for i=1:N
        for j=1:M
            Idx = knnsearch(TumourPoints{i}, (R*P(:,j,i)+t)');
            TCP(:,j,i)=(TumourPoints{i}(Idx,:))';
        end
    end
end

function normX = vec_norm(X)
    normX = sqrt(sum(X.^2));
end
