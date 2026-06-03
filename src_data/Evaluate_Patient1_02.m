close all
clear all

% ============================================================
% Evaluate_Patient1_02.m
% ------------------------------------------------------------
% 功能：
% 评估指定方法在指定 Patient / frame 的输出 STL。
% 默认配置为 FoundationPose 在 Patient1 / frame 02。
% 后续修改病例和帧数时，只需要改下面的配置区。
%
% 输入：
%   DATA/Registration Methods/{MethodName}/Patient{PatientNo}/{FrameNumber}.stl
%   DATA/Dataset/Patient{PatientNo}/LUS Calibration and Pose/{FrameNumber}.mat
%   DATA/Dataset/Patient{PatientNo}/LUS Segmentation/json/{FrameNumber}.json
%
% 输出：
%   DATA/Evaluation_Results/{MethodName}_Patient{PatientNo}_{FrameNumber}/{MethodName}_Patient{PatientNo}_{FrameNumber}.txt
%   DATA/Evaluation_Results/{MethodName}_Patient{PatientNo}_{FrameNumber}/{MethodName}_Patient{PatientNo}_{FrameNumber}.mat
%
% 直接在 MATLAB 中运行本脚本即可。
% ============================================================

ScriptDir = fileparts(mfilename('fullpath'));
ProjectRoot = fileparts(ScriptDir);
addpath(fullfile(ScriptDir, 'stlTools'));

% ---------------- User configuration ----------------
MethodName = 'FoundationPose';
PatientNo = 1;
FrameNumber = '02';
K = 10;              % ICP iterations for TRE
M = 50;              % number of samples over LUS tumour profile
OncologicMargin = 10; % mm, used for IC

DatasetPath = fullfile(ScriptDir, 'Dataset');
RegistrationRoot = fullfile(ScriptDir, 'Registration Methods');
PatientTag = ['Patient', num2str(PatientNo)];
EvalTag = [MethodName, '_', PatientTag, '_', FrameNumber];
ResultRoot = fullfile(ScriptDir, 'Evaluation_Results');
ResultPath = fullfile(ResultRoot, EvalTag);

if ~exist(ResultPath, 'dir')
    mkdir(ResultPath);
end

pathUSprj = fullfile(DatasetPath, PatientTag, 'LUS Calibration and Pose');
regPath = fullfile(RegistrationRoot, MethodName, PatientTag);
USSegPath = fullfile(DatasetPath, PatientTag, 'LUS Segmentation', 'json');

stlFile = fullfile(regPath, [FrameNumber, '.stl']);
matFile = fullfile(pathUSprj, [FrameNumber, '.mat']);
jsonFile = fullfile(USSegPath, [FrameNumber, '.json']);

generatedStlFile = fullfile(ProjectRoot, 'result', PatientTag, FrameNumber, ...
    'evaluation', 'Registration Methods', MethodName, PatientTag, [FrameNumber, '.stl']);
if ~exist(stlFile, 'file') && exist(generatedStlFile, 'file')
    if ~exist(regPath, 'dir')
        mkdir(regPath);
    end
    copyfile(generatedStlFile, stlFile);
    fprintf('[Info] Copied generated STL to evaluation input: %s\n', stlFile);
end

if ~exist(stlFile, 'file')
    error(['Missing registration STL: ', stlFile]);
end
if ~exist(matFile, 'file')
    error(['Missing LUS calibration/pose MAT: ', matFile]);
end
if ~exist(jsonFile, 'file')
    error(['Missing LUS segmentation JSON: ', jsonFile]);
end

fprintf('=== Evaluating %s Patient%d frame %s ===\n', MethodName, PatientNo, FrameNumber);
fprintf('STL : %s\n', stlFile);
fprintf('MAT : %s\n', matFile);
fprintf('JSON: %s\n\n', jsonFile);

[Points, Connectivity, n, name] = stlRead(stlFile);
% Original evaluation code contains the following optional conversion:
% Points(:,2:3) = -Points(:,2:3);
% It is intentionally kept disabled here to match the current Python STL export.

TumourPoints = {Points};
TumourConnectivity = {Connectivity};

fid = fopen(jsonFile);
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
data = jsondecode(str);

USprofile = getTumourProfileFromJson(data);
load(matFile);

USprofile = [SLus * USprofile(1,:); zeros(size(USprofile(1,:))); SAus * USprofile(2,:)];
USprofile_probe = (Rus * USprofile) + Tus;
USprofile_cam = (Rpr * USprofile_probe + Tpr);

P = zeros(3, M, 1);
P(:,:,1) = SampleProfile(USprofile_cam, M);

TRE = TREcalc(P, TumourPoints, K);
IC = ICcalc(P, TumourPoints, TumourConnectivity, OncologicMargin);

if IC
    ICText = 'Pass';
else
    ICText = 'Failed';
end

fprintf('Patient %d frame %s:\n', PatientNo, FrameNumber);
fprintf('TRE = %.6f mm\n', TRE);
fprintf('IC  = %s\n', ICText);

resultTxt = fullfile(ResultPath, [EvalTag, '.txt']);
fid = fopen(resultTxt, 'w');
fprintf(fid, 'Method: %s\n', MethodName);
fprintf(fid, 'Patient: %d\n', PatientNo);
fprintf(fid, 'Frame: %s\n', FrameNumber);
fprintf(fid, 'STL: %s\n', stlFile);
fprintf(fid, 'TRE_mm: %.10f\n', TRE);
fprintf(fid, 'IC: %s\n', ICText);
fprintf(fid, 'IC_numeric: %d\n', IC);
fclose(fid);

resultMat = fullfile(ResultPath, [EvalTag, '.mat']);
save(resultMat, 'TRE', 'IC', 'ICText', 'MethodName', 'PatientNo', 'FrameNumber', 'stlFile', 'jsonFile', 'matFile', 'P', 'TumourPoints', 'TumourConnectivity');

fprintf('\n[OK] Result TXT: %s\n', resultTxt);
fprintf('[OK] Result MAT: %s\n', resultMat);


function USprofile = getTumourProfileFromJson(data)
    % JSON structures may be decoded either as struct array or cell-like form depending on MATLAB version.
    if isfield(data, 'objects')
        objects = data.objects;
    else
        error('JSON does not contain objects field.');
    end

    tumourObj = [];
    if isstruct(objects)
        for idx = 1:numel(objects)
            obj = objects(idx);
            if isfield(obj, 'classTitle') && strcmpi(obj.classTitle, 'Tumor')
                tumourObj = obj;
                break;
            end
        end
        if isempty(tumourObj)
            tumourObj = objects(1);
        end
    else
        error('Unsupported JSON objects format.');
    end

    exterior = tumourObj.points.exterior;
    if iscell(exterior)
        pts = cell2mat(exterior);
    else
        pts = exterior;
    end
    USprofile = pts';
end


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
R0=eye(3);
t0=[];
for i=1:N
    CoG_P = mean(P(:,:,i),2);
    CoG_T = mean(TumourPoints{i}',2);
    t0 = [t0,CoG_T-CoG_P];
end
t0 = mean(t0,2);

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
